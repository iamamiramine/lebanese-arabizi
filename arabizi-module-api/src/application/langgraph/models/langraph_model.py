# langraph_model.py
from typing import Dict, Any
import logging
import traceback
import os
from datetime import datetime

# External dependencies
from langgraph.graph import StateGraph, START, END
# Embeddings are now handled by the global ChromaDB repository

from core.agents.generative_agent import GenerativeAgent
from core.agents.rag_agent import RAGAgent
from core.tools.vector_db_tool import VectorDBTool

# Internal application dependencies
from domain.models.states.graph_state import GraphState

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowGraph:
    """Main workflow graph that orchestrates all agents."""
    
    def __init__(self, 
        generator: GenerativeAgent, 
        rag_agent: RAGAgent, 
        vector_db_tool: VectorDBTool
    ):
        """Initialize the workflow graph with pre-initialized agents and tools."""        
        # Use pre-initialized agents and tools

        self.generator = generator
        self.rag_agent = rag_agent
        self.vector_db_tool = vector_db_tool

        self.graph = self.create_graph()

    def should_load_vector_db(self, state: GraphState):
        if not state["vector_db_loaded"]:
            return "vector_db_tool"
        return "generative_agent"


    def create_graph(self) -> StateGraph:
        """Create and configure the workflow graph."""
        # Create the graph
        workflow = StateGraph(GraphState)

        # Add nodes for each agent/component 
        workflow.add_node("generative_agent", self.generator)   
        workflow.add_node("rag_agent", self.rag_agent)
        workflow.add_node("vector_db_tool", self.vector_db_tool)

        # Workflow RAG
        workflow.add_edge(START, "rag_agent")
        workflow.add_conditional_edges("rag_agent", self.should_load_vector_db, ["vector_db_tool", "generative_agent"])
        workflow.add_edge("vector_db_tool", "rag_agent")  # Return to rag_agent after getting vector_db
        workflow.add_edge("generative_agent", END)

        # # Direct Workflow (No RAG)
        # workflow.add_edge(START, "generative_agent")
        # workflow.add_edge("generative_agent", END)
        
        # Compile the graph
        compiled_graph = workflow.compile()
        
        return compiled_graph
        
    def invoke(self, query: str) -> Dict[str, Any]:
        """Invoke the workflow with a query.
        
        Args:
            query: The user's question or query
            
        Returns:
            Dict[str, Any]: The final workflow state
        """
        logger.info(f"Invoking workflow with query: {query}")
            
        # Create initial state as a dictionary
        current_time = datetime.now()
        state_dict = {
            "query": query, 
            "source_path": "./data/books", 
            "collection_name": "chroma", 
            "vector_db_loaded": False,
            "context": None,
            "timestamp": current_time,
            "created_at": current_time
        }
        
        # Create and run the graph without tracing for now
        try:
            logger.info("Graph created, invoking workflow")
            final_state = self.graph.invoke(state_dict)
            logger.info("Workflow execution completed")
            return final_state
        except Exception as e:
            logger.error(f"Error invoking workflow: {e}")
            traceback.print_exc()  # TODO: Remove this
            return {"error": str(e), "generated_response": f"I encountered an error processing your request: {str(e)}"} 