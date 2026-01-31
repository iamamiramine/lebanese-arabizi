# generative_agent.py
from typing import Dict, Any, Optional
import logging
import json

from langchain_core.prompts.chat import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)

# 👉 Update the import to your new schema
# e.g., domain/models/structured_outputs/arabizi_chat_structured_output.py
from domain.models.structured_outputs.generative_agent_structured_output import ArabiziChatResponse
from infrastructure.services.memory_logger_service import get_memory_logger

logger = logging.getLogger(__name__)


def _safe_format_system_prompt(system_prompt: str, ctx: str | None) -> str:
    class _Safe(dict):
        def __missing__(self, key):  # unknown keys -> empty
            return ""
    return system_prompt.format_map(_Safe(context=(ctx or "").strip()))


def _parse_structured_response(raw) -> ArabiziChatResponse:
    """
    Parse the LLM output into ArabiziChatResponse.
    Supports: Pydantic object, dict, or JSON string from the LLM.
    """
    if isinstance(raw, ArabiziChatResponse):
        return raw

    if isinstance(raw, dict):
        return ArabiziChatResponse(**raw)

    if isinstance(raw, str):
        # Try strict JSON first
        try:
            data = json.loads(raw)
            return ArabiziChatResponse(**data)
        except Exception:
            pass

    # If the provider returns a tool-call-like object with .content or .additional_kwargs
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        try:
            return ArabiziChatResponse(**json.loads(content))
        except Exception:
            pass

    # Last resort: best-effort minimal object
    logger.warning("Falling back to minimal ArabiziChatResponse; could not parse structured output.")
    return ArabiziChatResponse(lebanese_arabizi_response="ma fhemet, feek t3ed phrasing?")


class GenerativeAgent:
    """Agent responsible for generating Arabizi chat responses using the LLM."""

    def __init__(self, llm, generation_config, chat_history, system_prompt, formatter):
        """
        llm: Any LangChain-compatible LLM with .invoke(...)
        generation_config: dict with provider settings (expects at least 'model_name')
        chat_history: an object with add_user_message/add_ai_message and iterable message store
        system_prompt: a template string (may include {context})
        formatter: an object with .format_chat_history(history, model_name)
        """
        self.llm = llm
        self.chat_history = chat_history
        self.system_prompt = system_prompt
        self.generation_config = generation_config
        self.formatter = formatter
        self.memory_logger = get_memory_logger()  # optional, but now actually used

    def format_chat_history(self, chat_history):
        """Format the chat history for different model types."""
        return self.formatter.format_chat_history(
            chat_history, self.generation_config.get("model_name", "")
        )

    def _build_llm_input(self, system_prompt: str, history_with_memory: str, user_query: str) -> Dict[str, Any]:
        """
        Prepare a provider-agnostic input payload.
        Many LangChain LLMs accept a dict; if yours needs a list of messages, adapt here.
        """
        return {
            "system": str(system_prompt),
            "history": history_with_memory,
            "user": user_query,
        }

    def _append_ai_messages(self, structured: ArabiziChatResponse):
        """Append Arabizi response and meta fields to chat history."""
        if structured.lebanese_arabizi_response:
            self.chat_history.add_ai_message(AIMessage(content=structured.lebanese_arabizi_response))

        meta_lines = []
        if structured.reasoning:
            meta_lines.append("Reasoning: " + " | ".join(structured.reasoning))

        if meta_lines:
            self.chat_history.add_ai_message(AIMessage(content="\n".join(meta_lines)))

    def generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response based on the current state."""
        query = state.get("query", "") or ""
        context = state.get("context", None)

        # Return early if no user query
        if not query.strip():
            return state

        try:
            # Store query for downstream tools
            state["query_to_process"] = query

            # Add user message to chat
            self.chat_history.add_user_message(HumanMessage(content=query))
            history_with_memory = self.format_chat_history(self.chat_history)

            # Prepare prompts
            system_prompt = _safe_format_system_prompt(self.system_prompt, context)

            print(f"System prompt: {system_prompt}", flush=True)

            # Invoke LLM
            llm_input = self._build_llm_input(system_prompt, history_with_memory, query)
            print(f"LLM input: {llm_input}", flush=True)
            raw_response = self.llm.invoke(llm_input)

            # Parse to structured schema
            structured_response = _parse_structured_response(raw_response)

            # Update chat history (Arabizi + concise metadata)
            self._append_ai_messages(structured_response)

            # Persist structured output in state for downstream agents
            state["generative_agent_response"] = structured_response.model_dump_json()

            # Log to memory (single consolidated record)
            try:
                self.memory_logger.log({
                    "component": "GenerativeAgent",
                    "model": self.generation_config.get("model_name"),
                    "query": query,
                    "context_present": context is not None,
                    "structured_output": structured_response.model_dump(),
                })
            except Exception as log_exc:
                logger.warning(f"Memory logging failed in GenerativeAgent: {log_exc}")

            return state

        except Exception as e:
            logger.exception("Error in GenerativeAgent.generate_response")
            state["generative_agent_response"] = (
                f"I encountered an error generating a response: {str(e)}"
            )
            # Still add a graceful fallback to chat history in Arabizi
            self.chat_history.add_ai_message(
                AIMessage(content="3tazaret shi khallani ma ehtamel el reqwest. Jarrib ba3den, law sama7et. 🙏")
            )
            return state

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response when the agent is called."""
        return self.generate_response(state)
