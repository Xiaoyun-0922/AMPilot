from typing import List, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Represents the state of the agent at any point in the conversation.

    Attributes:
        messages: A list of messages that make up the conversation history.
                  The last message in the list is the most recent one.
    """
    messages: List[BaseMessage]