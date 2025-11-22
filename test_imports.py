"""
Test script to verify all imports work correctly without circular dependencies.
Run this before running the main graph.py
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("="*60)
print("Testing Imports for Argo Float Project")
print("="*60)

# Test 1: Models
print("\n[1/5] Testing models...")
try:
    from src.state.models import ScientificIntent, FloatChatState
    print("✓ Models imported successfully")
    
    # Test creating a ScientificIntent
    test_intent = ScientificIntent(
        variable="temp",
        lat=19.0760,
        lon=72.8777,
        depth=1000,
        time_range=("2024-01-01", "2024-01-31")
    )
    print(f"✓ Created test ScientificIntent: {test_intent.variable}")
    
except Exception as e:
    print(f"✗ Error importing models: {e}")
    sys.exit(1)

# Test 2: GeoSolver
print("\n[2/6] Testing geosolver...")
try:
    from src.tools.geosolver import resolve_location_fast, GeoSolver
    print("✓ GeoSolver imported successfully")
except Exception as e:
    print(f"✗ Error importing geosolver: {e}")
    sys.exit(1)

# Test 3: Fetcher
print("\n[3/6] Testing fetcher...")
try:
    from src.tools.fetcher import fetch_argo_data
    print("✓ Fetcher imported successfully")
except Exception as e:
    print(f"✗ Error importing fetcher: {e}")
    sys.exit(1)

# Test 3: Intent Extractor
print("\n[3/5] Testing intent extractor...")
try:
    from src.tools.intent_extractor import extract_intent_with_llm
    print("✓ Intent extractor imported successfully")
except Exception as e:
    print(f"✗ Error importing intent extractor: {e}")
    sys.exit(1)

# Test 4: Graph
print("\n[4/5] Testing graph...")
try:
    from src.agent.graph import create_argo_workflow, run_argo_workflow
    print("✓ Graph imported successfully")
except Exception as e:
    print(f"✗ Error importing graph: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Dependencies
print("\n[5/5] Testing external dependencies...")
dependencies = {
    "pydantic": "Pydantic",
    "langchain": "LangChain",
    "langgraph": "LangGraph",
    "argopy": "ArgoPy",
    "pandas": "Pandas",
    "dotenv": "Python-dotenv"
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {name} available")
    except ImportError:
        print(f"✗ {name} NOT installed")
        missing.append(name)

if missing:
    print(f"\n⚠ Missing packages: {', '.join(missing)}")
    print("Install them with: pip install " + " ".join(missing))
    sys.exit(1)

# Final check
print("\n" + "="*60)
print("✓ All imports successful! Ready to run workflow.")
print("="*60)
print("\nYou can now run:")
print("  python src/agent/graph.py")
print("\nOr import in Python:")
print("  from src.agent.graph import run_argo_workflow")
print("  result = run_argo_workflow('your query here')")
print("="*60)