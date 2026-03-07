def route_problem(parsed_problem: dict) -> str:
    """
    Evaluates the parsed problem properties to route it.
    Returns the next node in the LangGraph sequence as a string.
    """
    if parsed_problem.get("needs_clarification", False) is True:
        return "human_in_the_loop"
    
    return "solver"
