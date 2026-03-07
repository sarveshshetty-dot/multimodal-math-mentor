import sys
import os

# Ensure we're running from the correct directory
sys.path.append(os.path.dirname(__file__))

from app import app_graph
import uuid

def test_pipeline():
    print("Initializing test...")
    # Generate a unique thread ID for memory
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Simple algebraic problem that should trigger parser -> router -> solver -> verifier -> explainer
    test_input = {"raw_text": "Solve for x: 3x - 7 = 14. Make sure to explain the steps."}
    
    print(f"Running pipeline with input: {test_input['raw_text']}")
    print("This may take a minute or two as TinyLlama runs on CPU...")
    
    try:
        final_state = app_graph.invoke(test_input, config=config)
        print("\n--- PIPELINE COMPLETED SUCCESSFULLY ---")
        print("\nParsed Problem:", final_state.get('parsed_problem'))
        print("\nSolution Steps:", final_state.get('solution_steps'))
        print("\nVerifier Output:", final_state.get('verifier_output'))
        print("\nFinal Explanation:\n", final_state.get('final_explanation'))
    except Exception as e:
        print("\n--- PIPELINE FAILED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pipeline()
