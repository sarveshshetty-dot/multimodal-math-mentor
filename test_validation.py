from agents.solver_agent import run_solver_agent

tests = [
    {'problem_text': 'j', 'topic': 'algebra'},
    {'problem_text': 'hello', 'topic': 'algebra'},
    {'problem_text': '2x + 5 = 13', 'topic': 'algebra'},
    {'problem_text': 'x**2', 'topic': 'algebra'},
    {'problem_text': 'derivative of x**2', 'topic': 'calculus'},
]

print("--- Validation Test ---\n")
for prob in tests:
    r = run_solver_agent(prob, '')
    print(f"Input: {prob['problem_text']}")
    print(f"  => Solution: {r['solution']}")
    print(f"  => Verified: {r['sympy_verified']}")
    print("-" * 30)
