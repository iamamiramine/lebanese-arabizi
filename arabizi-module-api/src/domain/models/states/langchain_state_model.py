class LangchainState:
    """
    Class to manage the state of Langchain components and resources.
    Tracks loading status of models, retrievers, chains and other resources.
    """
    def __init__(self):
        # Core model components
        self.generative_model = None
        self.generative_chain = None
        self.generator = None
        self.rag_agent = None
        self.vector_db_tool = None

    def is_model_loaded(self) -> bool:
        """Check if both model and tokenizer are loaded"""
        return self.generative_model is not None

    def is_pipeline_loaded(self) -> bool:
        """Check if HuggingFace pipeline is initialized"""
        return self.generative_chain is not None
