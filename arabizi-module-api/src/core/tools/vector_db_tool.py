from typing import Dict, Any, List, Type, Optional
import logging
import os
import pickle
import time
import json
from pathlib import Path
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain.tools import BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.retrievers import BM25Retriever

from infrastructure.repository.chromadb_repository import get_chromadb_repository, ChromaDBRepository

logger = logging.getLogger(__name__)


class VectorDBInput(BaseModel):
    source_path: str = Field(description="Path to directory containing Lebanese data sources")
    collection_name: str = Field(default="chroma", description="Name for the Chroma collection")


class VectorDBTool(BaseTool):
    """
    Improved vector DB tool for RAG — supports hybrid retrieval (dense + BM25),
    smart chunking, and automatic updates.
    """

    name: str = "vector_db_manager"
    description: str = "Manages vector databases and hybrid retrieval indices for Lebanese RAG."
    args_schema: Type[BaseModel] = VectorDBInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chromadb_repo: ChromaDBRepository = get_chromadb_repository()

    # ----------------------------------------------------------------------
    def _run(self, source_path: str, collection_name: str = "chroma") -> Dict[str, Any]:
        """Main entry — ensures vector DB is up-to-date."""
        # Always attempt to get/create the vector store and update it.
        main_store = self._chromadb_repo.get_main_vector_store()
        self._process_and_add_documents(source_path, collection_name)
        self._log_persist_dir(main_store)
        return {"vector_db_loaded": True}

    # ----------------------------------------------------------------------
    def _process_and_add_documents(self, source_path: str, collection_name: str = "chroma"):
        """Smart indexing with manifest check + hybrid retrieval prep."""
        try:
            manifest_path = Path("data/store/index_manifest.json")
            previous = {}
            if manifest_path.exists():
                try:
                    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("Manifest corrupt; reindexing entire dataset")

            current_files = self._list_supported_files(source_path)
            current_index = {str(p): p.stat().st_mtime for p in current_files}
            to_update = [p for p in current_files if str(p) not in previous or current_index[str(p)] > float(previous.get(str(p), 0))]

            if not to_update:
                logger.info("No new or modified documents detected; skipping indexing")
                return

            logger.info(f"Indexing {len(to_update)} new/updated documents...")
            docs = self._load_documents_for_paths(to_update)
            splits = self.split_documents(docs, chunk_size=700, chunk_overlap=150)
            valid = [d for d in splits if d.page_content.strip()]

            if not valid:
                logger.warning("No valid text chunks; aborting update")
                return

            store = self._chromadb_repo.get_main_vector_store()
            self._add_in_batches(store, valid)

            store.persist()
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(current_index, indent=2), encoding="utf-8")
            logger.info("Vector DB updated successfully")

            # Create auxiliary BM25 retriever cache
            self.save_text_chunks(valid)
            logger.info("BM25 text cache updated")

        except Exception as e:
            logger.error(f"VectorDBTool error: {e}")
            raise

    # ----------------------------------------------------------------------
    def split_documents(self, docs, chunk_size=700, chunk_overlap=150):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        contents = [doc.page_content if isinstance(doc, Document) else str(doc) for doc in docs]
        texts = splitter.create_documents(contents)
        logger.info(f"Split {len(texts)} text chunks.")
        return texts

    def save_text_chunks(self, texts, file_path: str = "./data/store/chunks.pkl"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(texts, f)

    def load_text_chunks(self, file_path: str = "./data/store/chunks.pkl"):
        if not os.path.exists(file_path):
            return []
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def _add_in_batches(self, store, docs: list, batch_size: int = 128):
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            try:
                store.add_documents(batch)
            except Exception as e:
                logger.warning(f"Retrying batch {i} due to {e}")
                time.sleep(1)
                store.add_documents(batch)

    def _list_supported_files(self, data_dir: str):
        exts = {".txt", ".csv", ".pdf", ".json"}
        return [p for p in Path(data_dir).rglob("*") if p.suffix.lower() in exts]

    def _load_documents_for_paths(self, paths: List[Path]):
        docs: List[Document] = []
        for path in paths:
            try:
                suffix = path.suffix.lower()
                if suffix == ".txt":
                    docs.extend(TextLoader(str(path)).load())
                elif suffix == ".pdf":
                    docs.extend(PyPDFLoader(path).load())
                elif suffix == ".csv":
                    docs.extend(CSVLoader(file_path=str(path)).load())
                elif suffix == ".json":
                    text = path.read_text(encoding="utf-8")
                    docs.append(Document(page_content=text))
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        return docs

    def _log_persist_dir(self, store):
        try:
            client = getattr(store, "_client", None)
            settings = getattr(client, "_settings", None)
            persist_dir = getattr(settings, "persist_directory", None)
            logger.info(f"Chroma persist directory: {persist_dir}")
        except Exception:
            logger.info("Could not log persist directory.")
