from agents.solver_agent import run_solver_agent

tests = [
    # Basic Equations
    ({'problem_text': 'Solve for x: 5x + 10 = 25', 'topic': 'algebra'}, 'algebra'),
    # System of Equations
    ({'problem_text': 'x + y = 10 and x - y = 2', 'topic': 'algebra'}, 'system'),
    # Calculus - Derivative
    ({'problem_text': 'derivative of x**2 + 3x + 5', 'topic': 'calculus'}, 'derivative'),
    # Calculus - Integral
    ({'problem_text': 'integral of sin(x)', 'topic': 'calculus'}, 'integral'),
    # Limits
    ({'problem_text': 'limit of sin(x)/x as x approaches 0', 'topic': 'calculus'}, 'limit'),
    # Matrices
    ({'problem_text': 'determinant of [[1, 2], [3, 4]]', 'topic': 'linear_algebra'}, 'matrix_det'),
    ({'problem_text': 'inverse of [[1, 5], [2, 3]]', 'topic': 'linear_algebra'}, 'matrix_inv'),
    # Simplification
    ({'problem_text': 'simplify (x + 1)**2 - (x - 1)**2', 'topic': 'algebra'}, 'simplify'),
]

print("--- Comprehensive Solver Test ---\n")
for prob, label in tests:
    r = run_solver_agent(prob, '')
    print(f"[{label.upper()}] Input: {prob['problem_text']}")
    print(f"  => Result: {r['solution']}")
    print("-" * 30)
