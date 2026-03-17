import json
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def run_parser_agent(raw_text: str) -> dict:
    """
    Fast rule-based parser to extract structured problem info from raw input.
    Intentionally avoids calling LocalLLM to prevent slow blocking on the CPU.
    LocalLLM is reserved for Solver and Explainer (the value-add steps).
    """
    text = raw_text.strip()
    
    # Detect topic via keyword matching
    topic = "algebra"  # default
    text_lower = text.lower()
    if any(k in text_lower for k in ["integral", "derivative", "differentiate", "limit", "dy/dx", "d/dx", "∫"]):
        topic = "calculus"
    elif any(k in text_lower for k in ["probability", "chance", "p(", "bayes", "event", "random"]):
        topic = "probability"
    elif any(k in text_lower for k in ["matrix", "determinant", "vector", "eigenvalue", "transpose"]):
        topic = "linear_algebra"
    elif any(k in text_lower for k in ["solve", "equation", "x", "y", "quadratic", "polynomial", "factor", "root"]):
        topic = "algebra"

    # Extract variables (single letters used as unknowns)
    variables = list(set(re.findall(r'\b([a-zA-Z])\b(?!\s*=)', text)))
    # Filter out common English words
    stop_words = {'a', 'I', 'A', 'x', 'y', 'z', 'n', 'k', 't'}
    variables = [v for v in variables if v in stop_words or v.islower()]
    # Always include obvious math vars
    for var in ['x', 'y', 'z', 'n']:
        if var in text and var not in variables:
            variables.append(var)
    variables = list(set(variables))[:5]  # Cap at 5

    # Extract constraints (look for inequalities, domain hints)
    constraints = []
    constraint_patterns = [
        r'(where\s+[\w\s<>=!]+)',
        r'(for\s+[\w\s<>=!]+)',
        r'(given\s+[\w\s<>=!]+)',
        r'([a-zA-Z]\s*[><=!]+\s*[\d\w]+)',
    ]
    for pat in constraint_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        constraints.extend(matches[:2])
    constraints = list(set(constraints))[:3]

    # Clean up problem text
    problem_text = re.sub(r'\s+', ' ', text).strip()

    # Detect ambiguity (needs_clarification)
    math_chars = re.compile(r'[0-9\+\-\*\/\^=\(\)∫]|sin|cos|tan|log|limit|derivative|integral|matrix|root|sqrt', re.IGNORECASE)
    needs_clarification = False
    if len(text) < 3 and not any(var in text for var in ['x', 'y', 'z']):
        needs_clarification = True
    elif not math_chars.search(text):
        needs_clarification = True

    return {
        "problem_text": problem_text,
        "topic": topic,
        "variables": variables,
        "constraints": constraints,
        "needs_clarification": needs_clarification
    }
