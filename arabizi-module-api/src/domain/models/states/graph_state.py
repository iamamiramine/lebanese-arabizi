from typing import Any, TypedDict

class GraphState(TypedDict, total=False):
    """
    Type definition for graph state used in the workflow.
    This is a TypedDict that defines the expected structure of the state dictionary.
    """
    # Input fields
    query: str                   # The user's original query
    generative_agent_response: str # The response from the generative agent
    thread_id: str               # Thread ID for MongoDB checkpointer
    query_to_process: str        # Query to process

    vector_db_loaded: bool       # Whether the main vector database is loaded
    context: str

    source_path: str
    collection_name: str
    
