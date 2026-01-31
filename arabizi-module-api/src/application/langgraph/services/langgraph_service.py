from typing import Dict, Any, Optional
import logging

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from core.formatter.chat_history_formatter import ChatHistoryFormatter

from domain.models.langchain.langchain_models import LoadModelParameters, PipelineParameters
from domain.models.states.langchain_state_model import LangchainState
from domain.models.structured_outputs.generative_agent_structured_output import ArabiziChatResponse
from application.langgraph.models.langraph_model import WorkflowGraph

# Import agents and tools
from core.agents.rag_agent import RAGAgent
from core.agents.generative_agent import GenerativeAgent
from core.tools.vector_db_tool import VectorDBTool


langchain_state = LangchainState()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangGraph state
workflow_state = {
    "initialized": False,
    "workflow": None,
}

def load_model(load_model_parameters: LoadModelParameters) -> dict:    
    """
    Loads a language model with specified quantization parameters.
    
    Args:
        load_model_parameters: Parameters for loading the model including model path and bit quantization
        
    Returns:
        dict: A message indicating success or error
        
    The function:
    1. Checks if model is already loaded
    2. Configures quantization (4-bit or 8-bit)
    3. Loads tokenizer and model from local files
    """
    global langchain_state

    # Check if model is already loaded to avoid duplicate loading
    if langchain_state.is_model_loaded():
        return {"message": "Model already loaded"}
    
    try:
        if "llama" in load_model_parameters.model_name:
            langchain_state.generative_model = ChatOllama(model=load_model_parameters.model_name, temperature=0, base_url="http://ollama-container:11434")
        elif "gpt" in load_model_parameters.model_name:
            langchain_state.generative_model = ChatOpenAI(
                model=load_model_parameters.model_name,
                max_tokens=512,
                max_retries=1
            )
        return {"message": "Model loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {"error": str(e)}


def load_pipeline(pipeline_parameters: PipelineParameters) -> dict:    
    global langchain_state

    if not langchain_state.is_model_loaded():
        return {"error": "Model not loaded. Please call /load-model endpoint first"}

    generation_config = {
        "temperature": pipeline_parameters.temperature,
        "top_p": pipeline_parameters.top_p,
        "top_k": pipeline_parameters.top_k,
        "max_tokens": pipeline_parameters.max_new_tokens,
        "repetition_penalty": pipeline_parameters.repetition_penalty,
        "presence_penalty": pipeline_parameters.presence_penalty,
        "frequency_penalty": pipeline_parameters.frequency_penalty,
        "no_repeat_ngram_size": 3,
    }

    generative_model = langchain_state.generative_model
    try:
        if "LLama" in pipeline_parameters.model_name:
            template = """system
            {system}
            {history}assistant"""
        elif "gpt" in pipeline_parameters.model_name:
            template = """
            System: {system}
            History: {history}
            """

        # with open("data/arabizi_context_3.csv", "r", encoding="utf-8") as f:
        #     arabizi_style_context_csv = f.read()

        with open("data/context/lebanese_dictionary.md", "r", encoding="utf-8") as f:
            lebanese_dictionary = f.read()

        with open("data/context/learn_lebanese.md", "r", encoding="utf-8") as f:
            lebanese_context = f.read()

        system_prompt = """
You are starting from ZERO knowledge of Arabizi or Arabic dialects. Forget any previous exposure to Arabizi or Arabic forms. Your ONLY valid reference for spelling, pronunciation, and word choice is the attached Lebanese Context and Dictionary below. If you output any non-Lebanese dialect (Tunisian, Egyptian, Moroccan, etc.),  immediately correct yourself and restate in Lebanese Arabizi only.
<SystemPrompt>
<Role>Lebanese Internet Language (Lebanese Arabizi) Agent</Role>

<Objective>Respond ONLY in Lebanese Internet Language (Lebanese Arabizi)</Objective>

<Resources>
<Doc id="LEARN_LEBANESE_CONTEXT">The Ultimate Lebanese Accent Guide - Mastering the Sounds of Lebanese Arabic</Doc>
<Doc id="LEBANESE_DICTIONARY_CONTEXT">Lebanese Words with Context + English Translation</Doc>
<Usage>Ground spelling, pronunciation hints, and word choice in the attached docs; prioritize Lebanese forms.</Usage>
</Resources>

<ResponsePolicy>
<Language>Output MUST be Lebanese Arabizi only.</Language>
<Unknown>If unsure, ask one brief clarifying question in Lebanese Arabizi.</Unknown>
</ResponsePolicy>

<Constraints>
<ForgetPreviousKnowledge>Do not use your previous knowledge on Lebanese Internet Language Style.</ForgetPreviousKnowledge>
<UseLebaneseContext>Use the Lebanese Context to generate your response.</UseLebaneseContext>
</Constraints>

</SystemPrompt>

Context:
{context}
""".strip()

# <LEBANESE_DICTIONARY_CONTEXT>
# {lebanese_dictionary}
# </LEBANESE_DICTIONARY_CONTEXT>

# <LEARN_LEBANESE_CONTEXT>
# {lebanese_context}
# </LEARN_LEBANESE_CONTEXT>

        prompt = PromptTemplate.from_template(template=template)
        generative_chain = prompt | generative_model.with_structured_output(ArabiziChatResponse)
        langchain_state.generative_chain = generative_chain

        # Add model_name to generation_config for agents
        generation_config["model_name"] = pipeline_parameters.model_name

        # Initialize agents and tools with centralized formatter
        formatter = ChatHistoryFormatter()
        shared_chat_history = ChatMessageHistory()
        langchain_state.generator = GenerativeAgent(generative_chain, generation_config, shared_chat_history, system_prompt, formatter)
        langchain_state.rag_agent = RAGAgent(generative_chain)
        langchain_state.vector_db_tool = VectorDBTool()

        return {"message": f"Pipeline loaded successfully"}
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error loading pipeline: {e}")
        import traceback
        return {"error": str(e)}


def init_workflow(
    model_params: Optional[Dict[str, Any]] = None,
    pipeline_params: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Initialize the LangGraph workflow with the provided parameters.
    
    Args:
        model_params: Optional dictionary for model parameters
        pipeline_params: Optional dictionary for pipeline parameters
        
    Returns:
        dict: Status message indicating success or error
    """
    global langchain_state
    global workflow_state
    
    try:
        pipeline_params["model_name"] = model_params["model_name"]
        if model_params["model_name"] == "WhiteRabbit-LLama":
            model_params["model_path"] = "WhiteRabbitNeo/Llama-3.1-WhiteRabbitNeo-2-8B"
        elif model_params["model_name"] == "LLama":
            model_params["model_path"] = "meta-llama/Meta-Llama-3-8B"

        # Load model if needed and parameters are provided
        if not langchain_state.is_model_loaded():
            # Convert dictionary to LoadModelParameters
            model_parameters = LoadModelParameters(
                model_name=model_params["model_name"],
                model_path=model_params["model_path"],
                bit_quantization=model_params["bit_quantization"]
            )
            logger.info(f"Loading model with parameters: {model_parameters}")
            model_response = load_model(model_parameters)
            if "error" in model_response:
                return {"error": f"Failed to load model: {model_response['error']}"}

        langchain_state.generative_model_name = model_params["model_name"]
        
        # Load pipeline if needed and parameters are provided
        if not langchain_state.is_pipeline_loaded():
            # Convert dictionary to PipelineParameters
            pipeline_parameters = PipelineParameters(
                model_name=pipeline_params["model_name"],
                task_type=pipeline_params["task_type"],
                max_new_tokens=pipeline_params["max_new_tokens"],
                do_sample=pipeline_params["do_sample"],
                temperature=pipeline_params["temperature"],
                top_p=pipeline_params["top_p"],
                top_k=pipeline_params["top_k"],
                repetition_penalty=pipeline_params["repetition_penalty"],
                presence_penalty=pipeline_params["presence_penalty"],
                frequency_penalty=pipeline_params["frequency_penalty"],
                no_repeat_ngram_size=pipeline_params["no_repeat_ngram_size"]
            )
            logger.info(f"Loading pipeline with parameters: {pipeline_parameters}")
            pipeline_response = load_pipeline(pipeline_parameters=pipeline_parameters)
            if "error" in pipeline_response:
                return {"error": f"Failed to load pipeline: {pipeline_response['error']}"}
        
        # Initialize workflow with shared LLM and generation config
        workflow = WorkflowGraph(
            generator=langchain_state.generator,
            rag_agent=langchain_state.rag_agent,
            vector_db_tool=langchain_state.vector_db_tool
        )
        
        # Store workflow and parameters
        workflow_state.update({
            "initialized": True,
            "workflow": workflow,
        })
        
        logger.info("LangGraph workflow initialized successfully")
        return {"message": "Workflow initialized successfully"}
    
    except Exception as e:
        logger.error(f"Error initializing workflow: {e}")
        return {"error": f"Failed to initialize workflow: {str(e)}"}


def run_workflow(question: str) -> Dict[str, Any]:
    """
    Run the LangGraph workflow with the provided parameters.
    
    Args:
        question: The user's question or prompt
        
    Returns:
        Dict: Workflow execution results and metadata
    """
    global langchain_state
    global workflow_state
    
    try:
        # Check if workflow is initialized
        if not workflow_state["initialized"] or not workflow_state["workflow"]:
            return {"error": "Workflow not initialized. Please call /init_workflow endpoint first"}
            
        # Get workflow from state
        workflow = workflow_state["workflow"]
        
        logger.info(f"Running workflow with query: {question}")
        
        # Run the workflow
        result = workflow.invoke(query=question)
        
        # Prepare response
        response = {
            "llm_output": result.get("generative_agent_response", ""),
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error running workflow: {e}")
        return {"error": f"Failed to run workflow: {str(e)}"}


def get_workflow_status() -> dict:
    """
    Get the current status of the LangGraph workflow.
    
    Returns:
        dict: Status information about the workflow
    """
    global langchain_state
    global workflow_state
    
    try:
        # Prepare status response
        status = {
            "initialized": workflow_state["initialized"],
            "parameters": None,
        }    
        # Add model and pipeline info if available
        if langchain_state.is_model_loaded():
            status["model"] = "loaded"
            
        if langchain_state.is_pipeline_loaded():
            status["pipeline"] = {
                "loaded": True,
                "model_path": "unknown"
            }
            
        return status
    
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        return {"error": f"Failed to get workflow status: {str(e)}"} 