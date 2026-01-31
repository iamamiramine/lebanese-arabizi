from pydantic import BaseModel, Field
from typing import List, Optional

class ArabiziChatResponse(BaseModel):
    """
    Structured output for Lebanese Arabizi agent responses.
    """

    lebanese_arabizi_response: str = Field(
        description="Final Lebanese Arabizi message shown to the user."
    )

    reasoning: Optional[List[str]] = Field(
        default=None,
        description="Step-by-step reasoning behind interpreting and answering the query."
    )