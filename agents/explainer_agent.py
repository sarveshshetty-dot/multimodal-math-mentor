import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def run_explainer_agent(parsed_problem: dict, solver_output: dict, verifier_output: dict) -> str:
    """
    Template-based explainer. Generates a clean, formatted explanation
    from SymPy solver results. No LLM used — instant response.
    """
    problem_text = parsed_problem.get("problem_text", "N/A")
    topic = parsed_problem.get("topic", "algebra").title()
    solution = solver_output.get("solution", "N/A")
    steps = solver_output.get("steps", [])
    sympy_verified = solver_output.get("sympy_verified", False)
    confidence = verifier_output.get("confidence_data", {}).get("confidence", 0)
    verification_summary = verifier_output.get("verification", {}).get("summary", "")
    errors_found = verifier_output.get("verification", {}).get("errors_found", [])

    # Format step-by-step section
    steps_md = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])

    # Confidence badge
    if confidence >= 90:
        conf_badge = f"🟢 **High Confidence** ({confidence}/100)"
    elif confidence >= 70:
        conf_badge = f"🟡 **Medium Confidence** ({confidence}/100)"
    else:
        conf_badge = f"🔴 **Low Confidence** ({confidence}/100) — Please verify manually."

    # Error notes
    error_section = ""
    if errors_found:
        error_lines = "\n".join([f"- ⚠️ {e}" for e in errors_found])
        error_section = f"\n\n### ⚠️ Notes\n{error_lines}"

    explanation = f"""## 📐 Solution: {topic}

**Problem:** {problem_text}

---

### 🔢 Step-by-Step Solution
{steps_md}

---

### ✅ Final Answer
> **{solution}**

---

### 🔍 Verification
{verification_summary}
{conf_badge}
{error_section}

---
*Solved using SymPy — a symbolic mathematics library. All computation is done locally on your device.*
"""
    return explanation
