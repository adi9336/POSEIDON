"""
Graph module for defining and running the Argo workflow.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# Import your existing components
from src.state.models import FloatChatState


class ArgoWorkflow:
    """Manages the Argo data processing workflow."""

    def __init__(self):
        self.graph = StateGraph(FloatChatState)
        self._setup_workflow()

    def _setup_workflow(self):
        """Define the workflow nodes and edges."""
        # Add nodes
        self.graph.add_node("intent_extractor", self._extract_intent)
        self.graph.add_node("data_retriever", self._retrieve_data)
        self.graph.add_node("data_processor", self._process_data)

        # Define the workflow
        self.graph.add_edge("intent_extractor", "data_retriever")
        self.graph.add_edge("data_retriever", "data_processor")
        self.graph.add_edge("data_processor", END)

        # Set the entry point
        self.graph.set_entry_point("intent_extractor")

        # Compile the graph
        self.workflow = self.graph.compile()

    def _extract_intent(self, state: FloatChatState) -> Dict[str, Any]:
        """Extract scientific intent from user query."""
        if not hasattr(state, "user_query") or not state.user_query:
            raise ValueError("No user query provided")

        # For now, return a simple intent
        # In a real implementation, you would use your intent extraction logic here
        return {"intent": {"query": state.user_query, "type": "temperature"}}

    def _retrieve_data(self, state: FloatChatState) -> Dict[str, Any]:
        """Retrieve data based on the extracted intent."""
        # This is a placeholder - replace with your actual data retrieval logic
        return {"raw_data": f"data_for_{state.intent['query']}"}

    def _process_data(self, state: FloatChatState) -> Dict[str, Any]:
        """Process the retrieved data."""
        # This is a placeholder - replace with your actual data processing logic
        return {"processed_data": f"processed_{state.raw_data}"}


def create_argo_workflow() -> ArgoWorkflow:
    """Create and return a new Argo workflow instance."""
    return ArgoWorkflow()


def run_argo_workflow(
    workflow: ArgoWorkflow, initial_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the Argo workflow with the given initial state.

    Args:
        workflow: The ArgoWorkflow instance to run
        initial_state: Dictionary containing the initial state

    Returns:
        The final state after running the workflow
    """
    # Convert the initial state to a FloatChatState object
    state = FloatChatState(**initial_state)

    # Run the workflow
    result = workflow.workflow.invoke(state)

    # Convert the result to a dictionary
    return result.dict() if hasattr(result, "dict") else result
