def calculate_confidence(verifier_steps: list, solution_found: bool, sympy_success: bool) -> dict:
    """
    Calculates a confidence score based on the outcome of the verifier checks and solver executions.
    If confidence is < 70, HITL should be triggered.
    """
    score = 100
    
    if not solution_found:
        score -= 40
        
    if not sympy_success:
        score -= 20
        
    # Check if verifier actually checked the solution against constraints / edge cases
    num_checks = len(verifier_steps)
    if num_checks == 0:
        score -= 25
    elif num_checks == 1:
        score -= 10
        
    # Cap score between 0 and 100
    score = max(0, min(100, score))
    
    return {
        "confidence": score,
        "needs_hitl": score < 70,
        "reason": f"Confidence score is {score}/100. Verification performed {num_checks} checks."
    }
