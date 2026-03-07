import sys
import os
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def run_verifier_agent(parsed_problem: dict, solver_output: dict) -> dict:
    """
    Fast rule-based verifier. Checks the solver's result using SymPy.
    No LLM used — instant verification.
    """
    solution = solver_output.get("solution", "")
    sympy_verified = solver_output.get("sympy_verified", False)
    solution_value = solver_output.get("solution_value")
    problem_text = parsed_problem.get("problem_text", "")

    errors_found = []
    confidence = 50
    summary = ""

    if sympy_verified and solution_value:
        # Re-verify: plug solution back into original equation
        try:
            transformations = (standard_transformations + (implicit_multiplication_application,))
            # Extract equation
            eq_match = re.search(
                r'([0-9a-zA-Z\s\+\-\*\/\^\(\)\.]+=[0-9a-zA-Z\s\+\-\*\/\^\(\)\.]+)',
                problem_text
            )
            if eq_match:
                eq_str = eq_match.group(1).strip()
                lhs_str, rhs_str = eq_str.split("=", 1)
                lhs = parse_expr(lhs_str.strip(), transformations=transformations)
                rhs = parse_expr(rhs_str.strip(), transformations=transformations)

                # Parse solution value and substitute
                sol_val = parse_expr(solution_value.strip("[]").split(",")[0].strip(), transformations=transformations)
                free_vars = list((lhs - rhs).free_symbols)
                if free_vars:
                    check = (lhs - rhs).subs(free_vars[0], sol_val)
                    result = sp.simplify(check)
                    if result == 0:
                        confidence = 98
                        summary = f"✅ Solution verified by substitution: result is 0 (correct)."
                    else:
                        confidence = 30
                        errors_found.append(f"Substitution check failed: got {result}, expected 0.")
                        summary = "❌ Substitution check indicates possible error."
                else:
                    confidence = 75
                    summary = "Solution computed, no free variables to verify."
            else:
                confidence = 80
                summary = "SymPy solved successfully. Could not re-verify (no equation pattern found)."
        except Exception as e:
            confidence = 70
            summary = f"SymPy solved successfully. Verification skipped ({e})."
    elif "Error" in solution or "No solution" in solution:
        confidence = 20
        errors_found.append("Solver reported an error or no solution found.")
        summary = "⚠️ Problem could not be solved symbolically."
    else:
        confidence = 60
        summary = "Solution generated. Manual verification recommended."

    needs_hitl = confidence < 50

    return {
        "verification": {
            "is_correct": confidence >= 70,
            "errors_found": errors_found,
            "summary": summary,
        },
        "confidence_data": {
            "confidence": confidence,
            "reason": summary,
            "needs_hitl": needs_hitl,
        }
    }
