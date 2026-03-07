import sys
import os
import json
import re

# Add project root to sys.path
sys.path.append(os.getcwd())

from agents.parser_agent import run_parser_agent
from agents.solver_agent import run_solver_agent
from agents.verifier_agent import run_verifier_agent
from rag.retriever import retrieve_context
from memory.memory_store import MemoryStore

def test_structure():
    print("1. [STRUCTURE CHECK]")
    folders = ['agents', 'input_processing', 'memory', 'rag', 'tools', 'ui']
    missing = [f for f in folders if not os.path.isdir(f)]
    if not missing:
        print("✅ All folders exist.")
    else:
        print(f"❌ Missing folders: {missing}")

def test_parser():
    print("\n2. [PARSER AGENT TEST]")
    test_cases = [
        "Solve for x: 2x + 5 = 13",
        "derivative of x**2 + 3x",
        "j"
    ]
    for text in test_cases:
        parsed = run_parser_agent(text)
        print(f"Input: {text}")
        print(f"Topic: {parsed.get('topic')}, Vars: {parsed.get('variables')}")
        # Schema check
        required = ["problem_text", "topic", "variables", "constraints", "needs_clarification"]
        missing = [r for r in required if r not in parsed]
        if not missing:
            print("  ✅ Schema valid")
        else:
            print(f"  ❌ Missing fields: {missing}")

def test_rag():
    print("\n3. [RAG PIPELINE TEST]")
    try:
        context = retrieve_context("algebra", k=1)
        if context:
            print("✅ RAG retrieved context.")
        else:
            print("⚠️ RAG returned empty context (check if data/faiss_index is populated).")
    except Exception as e:
        print(f"❌ RAG Error: {e}")

def test_solver():
    print("\n4. [MATH SOLVER & VALIDATION TEST]")
    test_cases = [
        {"problem": "Solve for x: 2x + 5 = 13", "topic": "algebra"},
        {"problem": "Derivative of x**2 + 3x", "topic": "calculus"},
        {"problem": "limit of sin(x)/x as x approaches 0", "topic": "calculus"},
        {"problem": "hello", "topic": "algebra"}
    ]
    for case in test_cases:
        parsed = {"problem_text": case['problem'], "topic": case['topic']}
        solved = run_solver_agent(parsed, "")
        print(f"Input: {case['problem']}")
        print(f"  Solution: {solved.get('solution')}")
        print(f"  Verified: {solved.get('sympy_verified')}")

def test_verifier():
    print("\n5. [VERIFIER AGENT TEST]")
    parsed = {"problem_text": "2x = 10", "topic": "algebra"}
    solver_out = {"solution": "x = 5", "sympy_verified": True, "solution_value": "5"}
    verified = run_verifier_agent(parsed, solver_out)
    print(f"Verified x=5 for 2x=10: Confidence {verified['confidence_data']['confidence']}")
    if verified['confidence_data']['confidence'] > 90:
        print("✅ Confidence correct.")
    else:
        print("❌ Confidence lower than expected.")

def run_audit():
    print("--- PROJECT FINAL AUDIT ---\n")
    test_structure()
    test_parser()
    test_rag()
    test_solver()
    test_verifier()
    print("\n--- AUDIT COMPLETE ---")

if __name__ == "__main__":
    run_audit()
