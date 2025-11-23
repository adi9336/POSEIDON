import os
import sys
from typing import Dict, List, Optional

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from langgraph
from langgraph.graph import StateGraph, END

# Import models first (no dependencies)
from src.state.models import FloatChatState, ScientificIntent

# Import tools after models
from src.tools.intent_extractor import extract_intent_with_llm
from src.tools.geosolver import resolve_location_fast
from src.tools.fetcher import fetch_argo_data
from src.tools.processor import process_data as processor_process_data


# ---------------------------------------------------------
# 1. Define the nodes
# ---------------------------------------------------------
def extract_intent(state: FloatChatState) -> FloatChatState:
    """Extract intent from user query."""
    print(f"\n🔍 Extracting intent from: {state.user_query}")
    
    try:
        intent = extract_intent_with_llm(state.user_query)
        print(f"✓ Intent extracted successfully")
        
        # Create a new state with the extracted intent
        return FloatChatState(
            user_query=state.user_query,
            intent=intent,
            dataset=state.dataset,
            erddap_url=state.erddap_url,
            raw_data=state.raw_data,
            processed=state.processed,
            final_answer=state.final_answer
        )
    except Exception as e:
        error_msg = f"Error extracting intent: {str(e)}"
        print(f"❌ {error_msg}")
        return FloatChatState(
            user_query=state.user_query,
            intent=None,
            dataset=state.dataset,
            erddap_url=state.erddap_url,
            raw_data=state.raw_data,
            processed=state.processed,
            final_answer=error_msg
        )


def fetch_data(state: FloatChatState) -> FloatChatState:
    """Fetch data based on extracted intent."""
    print(f"\n📊 Fetching data...")
    
    if not state.intent:
        error_msg = "No intent available to fetch data"
        print(f"❌ {error_msg}")
        return FloatChatState(
            user_query=state.user_query,
            intent=state.intent,
            dataset=state.dataset,
            erddap_url=state.erddap_url,
            raw_data=state.raw_data,
            processed=state.processed,
            final_answer=error_msg
        )
    
    try:
        # Fetch the data using the fetcher, passing the state to store the file path
        df = fetch_argo_data(state.intent, state=state)
        
        if df is not None and not df.empty:
            success_msg = f"✓ Data fetched successfully: {len(df)} rows, {len(df.columns)} columns"
            print(success_msg)
            
            # The fetcher has already updated state.raw_data with the file path
            return FloatChatState(
                user_query=state.user_query,
                intent=state.intent,
                dataset="ArgoFloats",
                erddap_url=state.erddap_url,
                raw_data=state.raw_data,  # This now contains the file path
                processed=state.processed,
                final_answer=success_msg
            )
        else:
            error_msg = "No data found for the specified parameters"
            print(f"⚠️ {error_msg}")
            return FloatChatState(
                user_query=state.user_query,
                intent=state.intent,
                dataset=state.dataset,
                erddap_url=state.erddap_url,
                raw_data=None,
                processed=state.processed,
                final_answer=error_msg
            )
            
    except Exception as e:
        error_msg = f"Error fetching data: {str(e)}"
        print(f"❌ {error_msg}")
        return FloatChatState(
            user_query=state.user_query,
            intent=state.intent,
            dataset=state.dataset,
            erddap_url=state.erddap_url,
            raw_data=state.raw_data,
            processed=state.processed,
            final_answer=error_msg
        )


# Process data function is now imported from src.tools.processor as processor_process_data


# ---------------------------------------------------------
# 2. Build the graph
# ---------------------------------------------------------
def create_argo_workflow():
    """Create the Argo data workflow graph."""
    print("\n🔧 Building workflow graph...")
    
    # Define the graph
    workflow = StateGraph(FloatChatState)
    
    # Add nodes
    workflow.add_node("extract_intent", extract_intent)
    workflow.add_node("fetch_data", fetch_data)
    workflow.add_node("process_data", processor_process_data)
    
    # Define the edges
    workflow.add_edge("extract_intent", "fetch_data")
    workflow.add_edge("fetch_data", "process_data")
    workflow.add_edge("process_data", END)
    
    # Set entry point
    workflow.set_entry_point("extract_intent")
    
    # Compile the graph
    print("✓ Workflow graph compiled")
    return workflow.compile()


# ---------------------------------------------------------
# 3. Run the workflow
# ---------------------------------------------------------
def run_argo_workflow(query: str) -> dict:
    """Run the Argo data workflow with the given query."""
    print("\n" + "="*60)
    print("🌊 Starting Argo Workflow")
    print("="*60)
    
    # Initialize the graph
    workflow = create_argo_workflow()
    
    # Create initial state
    initial_state = FloatChatState(
        user_query=query,
        intent=None,
        dataset=None,
        erddap_url=None,
        raw_data="",  # Initialize with empty string
        processed=None,
        final_answer=None
    )
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    
    # Convert to dict
    if hasattr(result, 'model_dump'):
        result_dict = result.model_dump()
    elif hasattr(result, 'dict'):
        result_dict = result.dict()
    else:
        result_dict = result
    
    print("\n" + "="*60)
    print("✅ Workflow Complete")
    print("="*60)
    
    return result_dict


# ---------------------------------------------------------
# 4. Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example queries
    test_queries = [
        
        "What's the salinity at 500m depth near Mumbai in January 2024?",
        "what is the temperature at 500m depth near Mumbai in January 2024?",
        
    ]
    
    for query in test_queries:
        print(f"\n\n{'='*70}")
        print(f"TESTING QUERY: {query}")
        print(f"{'='*70}")
        
        # Run the workflow
        result = run_argo_workflow(query)
        
        # Print results
        print(f"\n📋 RESULTS:")
        print(f"   Final Answer: {result['final_answer']}")
        
        if result.get('intent'):
            print(f"\n   Extracted Intent:")
            intent_dict = result['intent'] if isinstance(result['intent'], dict) else result['intent'].model_dump()
            for key, value in intent_dict.items():
                if value is not None:
                    print(f"      • {key}: {value}")
        
        if result.get('raw_data') is not None:
            df = result['raw_data']
            print(f"\n   Data Summary:")
            print(f"      • Shape: {df.shape if hasattr(df, 'shape') else 'N/A'}")
            if hasattr(df, 'columns'):
                print(f"      • Columns: {', '.join(df.columns)}")
            if hasattr(df, 'head'):
                print(f"\n   First 3 rows:")
                print(df.head(3).to_string(index=False))
        
        print("\n" + "="*70 + "\n")