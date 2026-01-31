import os
import json
import re
import uuid

import streamlit as st
import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def get_config():
    """Load API configuration from JSON file and return complete config"""
    try:
        config_path = os.path.join("shared", "config", "api_config.json")
        with open(config_path) as f:
            api_config = json.load(f)
        
        return {
            "BASE_URL": api_config["generative_module"],
            "LANGGRAPH_URL": f"{api_config['generative_module']}/langgraph"
        }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        st.error("Failed to load API configuration")
        return None

def init_session_state():
    """Initialize Streamlit session state variables"""
    # Initialize chat message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize workflow state
    if "workflow_initialized" not in st.session_state:
        st.session_state.workflow_initialized = False
        
    # Initialize model and pipeline parameters
    if "model_params" not in st.session_state:

        st.session_state.model_params = {
            "model_name": "gpt-4o-mini",
            "model_path": "",
            "bit_quantization": 8
        }

        # st.session_state.model_params = {
        #     "model_name": "LLama",
        #     "model_path": "models/LLama8b",
        #     "bit_quantization": 8
        # }
    
    # Default parameters for text generation pipeline
    if "pipeline_params" not in st.session_state:
        st.session_state.pipeline_params = {
            "max_new_tokens": 2048,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "task_type": "text-generation",
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "no_repeat_ngram_size": 3
        }
    
    # Track previous pipeline parameters to detect changes
    if "previous_pipeline_params" not in st.session_state:
        st.session_state.previous_pipeline_params = st.session_state.pipeline_params.copy()

def render_header():
    """Renders the application header with logo and title"""
    logo_path = os.path.join("shared", "assets", "Logo.png")
    st.image(logo_path, width=100)
    st.title("Lebanese Arabizi Agent")

def render_generation_controls():
    """Render controls for generation parameters in the sidebar"""
    with st.sidebar:
        # Add a subheader for the generation settings section
        st.subheader("Generation Settings")

        # Model selector - chooses between WhiteRabbit and gpt-4o-mini
        st.session_state.model_params["model_name"] = st.selectbox(
            "Model",
            ["gpt-4o-mini"],
            index=0,
            help="gpt-4o-mini: Fast response time"
        )

        st.session_state.model_params["bit_quantization"] = st.selectbox(
            "Bit Quantization",
            [8, 4],
            index=0,
            help="Bit quantization"
        )

        # Temperature slider - controls randomness vs determinism
        st.session_state.pipeline_params["temperature"] = st.slider(
            "Temperature", 0.0, 2.0, 
            st.session_state.pipeline_params["temperature"],
            help="Higher values make output more random, lower values more deterministic"
        )
        
        # Top P slider - implements nucleus sampling
        st.session_state.pipeline_params["top_p"] = st.slider(
            "Top P", 0.0, 1.0, 
            st.session_state.pipeline_params["top_p"],
            help="Nucleus sampling: limits cumulative probability of tokens considered"
        )
        
        # Top K slider - limits token consideration pool
        st.session_state.pipeline_params["top_k"] = st.slider(
            "Top K", 1, 100, 
            st.session_state.pipeline_params["top_k"],
            help="Limits the number of tokens considered for each step"
        )
        
        # Max New Tokens slider - controls response length
        st.session_state.pipeline_params["max_new_tokens"] = st.slider(
            "Max New Tokens", 256, 4096, 
            st.session_state.pipeline_params["max_new_tokens"],
            help="Maximum length of generated response (excluding prompt)"
        )
        
        # Repetition Penalty slider - controls token reuse
        st.session_state.pipeline_params["repetition_penalty"] = st.slider(
            "Repetition Penalty", 1.0, 2.0, 
            st.session_state.pipeline_params["repetition_penalty"],
            help="Higher values reduce repetition in generated text"
        )

        # Presence Penalty slider - controls presence of tokens in generated text
        st.session_state.pipeline_params["presence_penalty"] = st.slider(
            "Presence Penalty", 0.0, 2.0, 
            st.session_state.pipeline_params["presence_penalty"],
            help="Higher values increase presence of tokens in generated text"
        )

        # Frequency Penalty slider - controls frequency of tokens in generated text
        st.session_state.pipeline_params["frequency_penalty"] = st.slider(
            "Frequency Penalty", 0.0, 2.0, 
            st.session_state.pipeline_params["frequency_penalty"],
            help="Higher values increase frequency of tokens in generated text"
        )

        # No Repeat N-Gram Size slider - controls size of n-grams to prevent repetition
        st.session_state.pipeline_params["no_repeat_ngram_size"] = st.slider(
            "No Repeat N-Gram Size", 1, 10, 
            st.session_state.pipeline_params["no_repeat_ngram_size"],
            help="Prevents repetition of n-grams of this size"
        )
        
        # Task Type selector - chooses between text generation and text2text generation
        task_type = st.radio(
            "Task Type",
            ["text-generation", "text2text-generation"],
            index=0 if st.session_state.pipeline_params.get("task_type", "text-generation") == "text-generation" else 1,
            help="text-generation: Regular text generation, text2text-generation: Text-to-text generation"
        )
        st.session_state.pipeline_params["task_type"] = task_type

def handle_pipeline_switch():
    """Handle pipeline switch with proper feedback"""
    current_params = {k: v for k, v in st.session_state.pipeline_params.items()}
    
    # Check if parameters have changed from previous state
    if current_params != st.session_state.previous_pipeline_params:
        # Update previous parameters
        st.session_state.previous_pipeline_params = current_params.copy()
        # Trigger workflow reinitialization when parameters change
        st.session_state.workflow_initialized = False

def initialize_workflow():
    """Initialize the LangGraph workflow"""
    config = get_config()
    if not config:
        return False

    try:
        # Create parameters for workflow initialization
        workflow_parameters = {
            "model_params": st.session_state.model_params,
            "pipeline_params": st.session_state.pipeline_params,
        }
        
        logger.info(f"Initializing workflow with model: {st.session_state.model_params['model_name']}")
        
        response = requests.post(
            f"{config['LANGGRAPH_URL']}/init_workflow", 
            json=workflow_parameters
        )
        
        if not response.ok:
            st.error(f"Failed to initialize workflow: {response.json().get('detail', 'Unknown error')}")
            return False
        
        st.success("Workflow initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing workflow: {e}")
        st.error(f"Error initializing workflow: {str(e)}")
        return False

def get_workflow_status():
    """Get the current status of the LangGraph workflow"""
    config = get_config()
    if not config:
        return None

    try:
        response = requests.get(f"{config['LANGGRAPH_URL']}/workflow_status")
        if not response.ok:
            st.error(f"Failed to get workflow status: {response.json().get('detail', 'Unknown error')}")
            return None
        
        return response.json()
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        st.error(f"Error getting workflow status: {str(e)}")
        return None

def run_workflow(prompt: str):
    """Run the LangGraph workflow with the given prompt"""
    config = get_config()
    if not config:
        return {"error": "Unable to load API configuration"}
    
    try:        
        params = {
            "question": prompt,
        }
        
        response = requests.post(
            f"{config['LANGGRAPH_URL']}/run_workflow", 
            json=params
        )

        if not response.ok:
            error_detail = response.json().get('detail', 'Unknown error')
            logger.error(f"Error from API: {error_detail}")
            return {"error": f"Error generating response: {error_detail}"}

        return response.json()
        
    except Exception as e:
        logger.error(f"Error running workflow: {e}")
        return {"error": f"Error running workflow: {str(e)}"}

def display_chat_message(message: dict):
    """Display chat message with proper formatting"""
    if message.get("format") == "code":
        st.code(message["content"], language="bash")
    else:
        st.markdown(message["content"], unsafe_allow_html=True)

def process_batch_query(query: str, query_index: int):
    """Process a single query from the batch"""
    try:
        # Add query to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": f"[Batch Query {query_index + 1}] {query}",
            "format": "text"
        })
        
        # Run the workflow
        response = run_workflow(query)
        
        if "error" in response:
            error_msg = f"[Batch Query {query_index + 1} Error] {response['error']}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "format": "text"
            })
            return {"success": False, "error": response["error"]}
        else:
            # Process successful response
            generated_response = response.get("llm_output", "")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"[Batch Query {query_index + 1} Response] {generated_response}",
                "format": "text"
            })
            
            return {
                "success": True, 
                "response": generated_response,
                "full_response": response
            }
            
    except Exception as e:
        error_msg = f"[Batch Query {query_index + 1} Exception] {str(e)}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "format": "text"
        })
        return {"success": False, "error": str(e)}

def main():
    """Main function that handles the Streamlit chat interface and interaction flow"""
    init_session_state()
    render_header()
    
    # Render the generation controls in sidebar
    render_generation_controls()
    
    # Handle pipeline parameter changes
    handle_pipeline_switch()

    # Check workflow status
    workflow_status = get_workflow_status()
    
    # Initialize workflow if not already initialized
    if not st.session_state.workflow_initialized or workflow_status is None or not workflow_status.get("initialized", False):
        with st.spinner("Initializing LangGraph workflow..."):
            if initialize_workflow():
                st.session_state.workflow_initialized = True
            else:
                st.error("Failed to initialize LangGraph workflow")
                st.stop()
    
    # Display workflow status in sidebar
    with st.sidebar:
        st.subheader("Workflow Status")
        if workflow_status:
            for key, value in workflow_status.items():
                if isinstance(value, dict):
                    st.write(f"**{key}:**")
                    for sub_key, sub_value in value.items():
                        st.write(f"  - {sub_key}: {sub_value}")
                else:
                    st.write(f"**{key}:** {value}")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=os.path.join("shared", "assets", "botpic.jpg") if message["role"] == "assistant" else None):
            display_chat_message(message)

    # Handle user input (disabled during batch processing)
    if prompt := st.chat_input("Ask me anything!"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "format": "text"
        })

        # Generate and display assistant response
        with st.spinner("Running LangGraph workflow..."):
            response = run_workflow(prompt)
            
            try:
                if "error" in response:
                    with st.chat_message("assistant", avatar=os.path.join("shared", "assets", "botpic.jpg")):
                        st.error(response["error"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["error"],
                        "format": "text"
                    })
                else:
                    # Process the generated response using the correct field name
                    generated_response = response.get("llm_output", "")
                    execution_status = response.get("execution_status", [])
                    
                    # Display the generated response
                    with st.chat_message("assistant", avatar=os.path.join("shared", "assets", "botpic.jpg")):
                        st.markdown(generated_response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": generated_response,
                        "format": "text"
                    })
                    
                    # Extract any code blocks
                    code_blocks = re.findall(r'```(?:bash|sh)\n(.*?)\n```', generated_response, re.DOTALL)
                    if code_blocks and execution_status:
                        # Display execution results if a script was executed
                        if execution_status and execution_status[0].get("executed_on_kali"):
                            execution_result = execution_status[0].get("execution_result", {})
                            
                            # Create a formatted output string
                            output_parts = []
                            if execution_result.get("output"):
                                output_parts.append("=== Command Output ===\n" + execution_result["output"])
                            if execution_result.get("error"):
                                output_parts.append("=== Error Output ===\n" + execution_result["error"])
                            
                            if output_parts:
                                kali_output = "\n".join(output_parts)
                                with st.chat_message("assistant", avatar=os.path.join("shared", "assets", "botpic.jpg")):
                                    st.markdown("**Kali Linux Output:**")
                                    st.code(kali_output, language="bash")
                                
                                # Store the Kali output with proper formatting
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": kali_output,
                                    "format": "code"
                                })
                    
            except Exception as e:
                error_msg = f"Error processing response: {str(e)}"
                logger.error(f"Unexpected error: {e}")
                with st.chat_message("assistant", avatar=os.path.join("shared", "assets", "botpic.jpg")):
                    st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "format": "text"
                })

if __name__ == "__main__":
    main() 