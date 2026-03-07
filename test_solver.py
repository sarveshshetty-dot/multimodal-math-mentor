from agents.solver_agent import run_solver_agent

tests = [
    ({'problem_text': 'What is the derivative of x**2?', 'topic': 'calculus', 'variables': ['x']}, 'derivative'),
    ({'problem_text': 'Find the derivative of sin(x)', 'topic': 'calculus', 'variables': ['x']}, 'derivative'),
    ({'problem_text': 'Solve for x: 3x - 7 = 14', 'topic': 'algebra', 'variables': ['x']}, 'equation'),
    ({'problem_text': 'Solve for x: 2x + 5 = 13', 'topic': 'algebra', 'variables': ['x']}, 'equation'),
    ({'problem_text': 'integral of x**2', 'topic': 'calculus', 'variables': ['x']}, 'integral'),
    ({'problem_text': 'x**2 - 4 = 0', 'topic': 'algebra', 'variables': ['x']}, 'quadratic'),
]
for prob, label in tests:
    r = run_solver_agent(prob, '')
    print(f"[{label}] {prob['problem_text']}")
    print(f"  => {r['solution']}")
