from typing import List, Dict, Any
import logging, random
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from infrastructure.repository.chromadb_repository import get_chromadb_repository

logger = logging.getLogger(__name__)


class MultiRetriever:
    """
    Combine multiple retrievers (e.g., Chroma + BM25) with static weighting.
    """
    def __init__(self, retrievers: list, weights: list[float]):
        assert len(retrievers) == len(weights)
        self._retrievers = retrievers
        self._weights = weights

    def get_relevant_documents(self, query: str) -> List[Document]:
        combined = []
        for r, w in zip(self._retrievers, self._weights):
            try:
                # Prefer Runnable retrievers in new LangChain versions
                if hasattr(r, "invoke"):
                    docs = r.invoke(query)
                else:
                    docs = r.get_relevant_documents(query)
                for d in docs:
                    combined.append((d, w))
            except Exception as e:
                logger.warning(f"Retriever error: {e}")
        # De-dupe + sort by weighted score
        seen, final = {}, []
        for doc, w in combined:
            key = doc.page_content.strip()[:300]
            if key not in seen or seen[key] < w:
                seen[key] = w
        for content, w in seen.items():
            final.append((Document(page_content=content), w))
        random.shuffle(final)
        return [d for d, _ in sorted(final, key=lambda x: -x[1])]


class RAGAgent:
    """
    Hybrid RAG agent with:
    - Dense + lexical retrieval (Chroma + BM25)
    - MMR diversification
    - Query expansion for transliteration variants
    - LLM-based reranking
    - Compact contextualization
    """

    def __init__(self, llm):
        self.llm = llm
        self.retriever = None
        self.bm25 = None

    # ----------------------------------------------------------------------
    def _build_retriever(self):
        """Build a hybrid retriever with MMR and BM25 fallback."""
        chromadb_repo = get_chromadb_repository()

        retrievers, weights = [], []

        # Dense retriever (Chroma)
        main_store = chromadb_repo.get_main_vector_store()
        if main_store:
            retrievers.append(
                main_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 10, "fetch_k": 25, "lambda_mult": 0.6},
                )
            )
            weights.append(0.7)

        # Lexical retriever (BM25)
        try:
            from core.tools.vector_db_tool import VectorDBTool
            bm25_docs = VectorDBTool().load_text_chunks()
            if bm25_docs:
                self.bm25 = BM25Retriever.from_documents(bm25_docs)
                retrievers.append(self.bm25)
                weights.append(0.3)
        except Exception as e:
            logger.warning(f"BM25 retriever init failed: {e}")

        if not retrievers:
            return None
        if len(retrievers) == 1:
            return retrievers[0]
        return MultiRetriever(retrievers, weights)

    # ----------------------------------------------------------------------
    def initialize_chain(self):
        """Initialize retrieval pipeline for LangGraph integration."""
        self.retriever = self._build_retriever()
        if not self.retriever:
            return None

        def _extract_question(raw: Any) -> str:
            try:
                if isinstance(raw, str):
                    return raw.strip()
                if isinstance(raw, dict):
                    inner = raw.get("question")
                    if isinstance(inner, str):
                        return inner.strip()
                    return (str(inner) if inner is not None else "").strip()
                return (str(raw) if raw is not None else "").strip()
            except Exception:
                return ""

        def enrich(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            q = _extract_question(input_dict.get("question"))
            if not q:
                return {"context": "", "retrieved_documents": []}

            # # Expand query for transliteration and synonyms
            # queries = self._expand_query(q)
            # docs = []
            # for subq in queries:
            #     sub_docs = self._call_child_retriever(subq)
            #     docs.extend(sub_docs)

            docs = self._call_child_retriever(q)
            docs = self._dedupe_docs(docs)
            # docs = self._rerank_with_llm(docs, q)
            formatted = self._format_docs(docs[:8])
            return {"context": formatted, "retrieved_documents": docs}

        return {"question": RunnablePassthrough()} | RunnableLambda(enrich)

    # ----------------------------------------------------------------------
    def _call_child_retriever(self, query: str):
        """Compatibility wrapper for retrievers."""
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(query)
        elif hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(query)
        return []

    # def _expand_query(self, query: str) -> List[str]:
    #     """Use LLM to generate paraphrases and transliteration variants."""
    #     prompt = PromptTemplate.from_template(
    #         "Generate 3 short paraphrases or transliteration variants of this Lebanese query:\n{query}"
    #     )
    #     try:
    #         chain = LLMChain(llm=self.llm, prompt=prompt)
    #         out = chain.run(query)
    #         variants = [query] + [l.strip() for l in out.split("\n") if l.strip()]
    #         return variants[:4]
    #     except Exception:
    #         return [query]

    def _dedupe_docs(self, docs: List[Document]):
        """Remove duplicate or near-identical documents."""
        seen, deduped = set(), []
        for d in docs:
            key = d.page_content.strip()[:200]
            if key not in seen:
                seen.add(key)
                deduped.append(d)
        random.shuffle(deduped)
        return deduped

    def _rerank_with_llm(self, docs: List[Document], query: str):
        """Optional LLM-based reranking for higher relevance."""
        if not docs or not self.llm:
            return docs
        scored = []
        for d in docs[:12]:
            snippet = d.page_content[:500]
            prompt = f"Rate the relevance (0 to 1) of this text to the Lebanese query '{query}':\n{snippet}"
            try:
                score = float(self.llm.invoke(prompt).content.strip())
            except Exception:
                score = 0.5
            scored.append((d, score))
        scored.sort(key=lambda x: -x[1])
        return [d for d, _ in scored]

    def _format_docs(self, docs: List[Document]):
        """Format retrieved documents into a compact context block."""
        ctx = []
        for i, d in enumerate(docs, 1):
            seg = f"[{i}] {d.page_content.strip()}\n"
            ctx.append(seg)
        return "\n".join(ctx)

    # ----------------------------------------------------------------------
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve and inject context into LangGraph state."""
        query = state.get("query", "")
        chain = self.initialize_chain()
        if chain:
            enriched = chain.invoke({"question": query})
            state["context"] = enriched.get("context", "")
        else:
            state["context"] = ""
        return state
