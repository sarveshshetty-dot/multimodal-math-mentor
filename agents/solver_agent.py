import sys
import os
import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

TRANSFORMATIONS = (standard_transformations + (implicit_multiplication_application, convert_xor))

# Natural language mappings for robust extraction
NL_MAP = {
    r'\^': '**',
    r'\bx squared\b': 'x**2',
    r'\by squared\b': 'y**2',
    r'\bz squared\b': 'z**2',
    r'\bx cubed\b': 'x**3',
    r'\bsquare\b': '**2',
    r'\bcubed\b': '**3',
    r'\btimes\b': '*',
    r'\bdivided by\b': '/',
    r'\bplus\b': '+',
    r'\bminus\b': '-',
    r'\bsin\s+of\b': 'sin',
    r'\bcos\s+of\b': 'cos',
    r'\btan\s+of\b': 'tan',
    r'\blog\s+of\b': 'log',
    r'\blimit\s+of\b': 'limit',
    r'\bas\s+x\s+approaches\b': ',',
    r'\bx\s*->\s*': ',',
    r'\bx\s*→\s*': ',',
}

# Prefixes to strip
PREFIX_STRIP = re.compile(
    r'^(solve|find|calculate|evaluate|what is|simplify|determine|compute)'
    r'(\s+for\s+[a-zA-Z](\s+and\s+[a-zA-Z])?\s*:?|\s*:\s*|\s+the\s+|\s+)',
    re.IGNORECASE
)

def _strip_prefix(text: str) -> str:
    return PREFIX_STRIP.sub('', text.strip())

def _normalize_nl(text: str) -> str:
    t = text.strip()
    for pat, rep in NL_MAP.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return t

def _extract_sympy_expr(raw: str) -> str:
    raw = raw.strip()
    # Handle "of" in keywords
    m = re.search(r'\bof\b\s+(.+?)$', raw, re.IGNORECASE)
    if m:
        return _normalize_nl(m.group(1))
    return _normalize_nl(raw)

def _detect_variables(expr) -> list:
    if hasattr(expr, "free_symbols"):
        return sorted(list(expr.free_symbols), key=lambda s: str(s))
    return []

def run_solver_agent(parsed_problem: dict, retrieved_context: str) -> dict:
    problem_text = parsed_problem.get("problem_text", "")
    topic = parsed_problem.get("topic", "algebra").lower()
    steps = []
    sympy_solution_str = None
    solution_value = None

    # --- Problem Classification & Graphing ---
    problem_type = parsed_problem.get("problem_type", "EXPRESSION")
    
    if problem_type == "FUNCTION":
        steps.append(f"**Classification:** Function (Graphing Mode)")
        try:
            # Extract RHS if "y =" exists
            rhs = problem_text
            if "y =" in problem_text.lower():
                rhs = problem_text.split("=", 1)[1].strip()
            elif re.match(r'^y\s*=', problem_text, re.IGNORECASE):
                rhs = re.sub(r'^y\s*=', '', problem_text, flags=re.IGNORECASE).strip()
            
            # Normalize and parse
            expr_str = _normalize_nl(rhs)
            expr = parse_expr(expr_str, transformations=TRANSFORMATIONS)
            
            # Generate Graph
            import matplotlib.pyplot as plt
            import numpy as np
            import tempfile
            
            x_vals = np.linspace(-10, 10, 400)
            # Create a lambda function for evaluation
            # Handle symbols robustly
            f = sp.lambdify(sp.Symbol('x'), expr, modules=['numpy', 'sympy'])
            y_vals = f(x_vals)
            
            # If y_vals is a single value (constant), broadcast it
            if np.isscalar(y_vals):
                y_vals = np.full_like(x_vals, y_vals)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_vals, y_vals, label=f"y = {expr}")
            ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)
            ax.axvline(0, color='black', linewidth=0.8, alpha=0.3)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_title(f"Graph of y = {expr}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=tempfile.gettempdir()) as tmp:
                plt.savefig(tmp.name)
                graph_path = tmp.name
            plt.close(fig)
            
            steps.append(f"**Parsed Expression:** `{expr}`")
            steps.append(f"**Graphing:** Generated graph in memory.")
            
            return {
                "solution": "This looks like a function. Displaying graph instead.",
                "steps": steps,
                "sympy_verified": True,
                "solution_value": str(expr),
                "is_graph": True,
                "graph_path": graph_path
            }
        except Exception as e:
            steps.append(f"**Graphing Error:** {e}")
            problem_type = "EQUATION" # Fallback
    
    steps.append(f"**Topic:** {topic.title()}")

    tl = problem_text.lower()

    # ─── VALIDATION ──────────────────────────────────────────────────
    # Check if the input is too short or lacks math characters
    math_chars = re.compile(r'[0-9\+\-\*\/\^=\(\)∫]|sin|cos|tan|log|limit|derivative|integral|matrix|root|sqrt', re.IGNORECASE)
    if len(problem_text.strip()) < 2 or not math_chars.search(problem_text):
        if len(problem_text.strip()) == 1 and problem_text.strip().isalpha():
            sympy_solution_str = f"'{problem_text}' is not a math problem. Please enter an equation or expression."
            steps.append(f"**Note:** Single letter input detected. Not a math problem.")
            return {
                "solution": sympy_solution_str,
                "steps": steps,
                "sympy_verified": False,
                "solution_value": None,
            }

    # ─── LIMITS ──────────────────────────────────────────────────────
    if 'limit' in tl:
        # Improved Limit Detection: "limit sin(x)/x as x -> 0"
        try:
            # 1. Strip the "limit" word from anywhere in the text
            text_clean = re.sub(r'\blimit\b\s*(of\b)?', '', problem_text, flags=re.IGNORECASE).strip()
            
            # 2. Find the approach part (as x approaches 0, x->0, etc.)
            # Normalization already maps x-> and "as x approaches" to "," 
            text_norm = _normalize_nl(text_clean)
            
            if ',' in text_norm:
                parts = [p.strip() for p in text_norm.split(',')]
                # Usually: [expression, limit_value]
                expr_str = parts[0]
                limit_val_str = parts[1]
                
                # Handle cases like "limit x->0 sin(x)/x" where normalize made it ",0 sin(x)/x"
                if not expr_str and limit_val_str:
                    # Input was like "limit x->0 ..." -> ",0 ..."
                    # Now parts[1] contains "0 sin(x)/x"
                    match = re.match(r'^([\d\.\-]+)\s+(.+)$', limit_val_str)
                    if match:
                        limit_val_str = match.group(1)
                        expr_str = match.group(2)

                expr = parse_expr(expr_str, transformations=TRANSFORMATIONS)
                limit_val = parse_expr(limit_val_str, transformations=TRANSFORMATIONS)
                var = _detect_variables(expr)[0] if _detect_variables(expr) else sp.Symbol('x')
                
                result = sp.limit(expr, var, limit_val)
                sympy_solution_str = f"lim_{{{var}→{limit_val}}} ({expr}) = {result}"
                steps.append(f"**Limit of:** `{expr}` as `{var} → {limit_val}`")
                steps.append(f"**Final Answer:** `{sympy_solution_str}` ✅")
                solution_value = [result]
            else:
                steps.append("**Note:** Please specify the approach value (e.g., 'as x->0').")
        except Exception as e:
            steps.append(f"**Error parsing limit:** {e}")

    # ─── MATRICES ────────────────────────────────────────────────────
    elif 'matrix' in tl or '[' in tl:
        # Very basic matrix detection: "determinant of [[1,2],[3,4]]"
        try:
            mat_match = re.search(r'\[\[.*?\]\]', problem_text)
            if mat_match:
                mat_str = mat_match.group(0)
                mat = sp.Matrix(eval(mat_str))
                if 'determinant' in tl:
                    det = mat.det()
                    sympy_solution_str = f"Determinant = {det}"
                    steps.append(f"**Matrix:** `{mat}`")
                    steps.append(f"**Computing Determinant...**")
                    steps.append(f"**Final Answer:** `{det}` ✅")
                    solution_value = [det]
                elif 'inverse' in tl:
                    inv = mat.inv()
                    sympy_solution_str = f"Inverse = {inv}"
                    steps.append(f"**Matrix:** `{mat}`")
                    steps.append(f"**Computing Inverse...**")
                    steps.append(f"**Final Answer:** `{inv}` ✅")
                    solution_value = [inv]
            else:
                steps.append("**Note:** Enter matrices in format `[[1,2],[3,4]]`.")
        except Exception as e:
            steps.append(f"**Error with matrix:** {e}")

    # ─── DERIVATIVE ──────────────────────────────────────────────────
    elif any(k in tl for k in ['derivative', 'differentiate', 'dy/dx', 'd/dx']):
        expr_str = _extract_sympy_expr(problem_text)
        try:
            expr = parse_expr(expr_str, transformations=TRANSFORMATIONS)
            vars_found = _detect_variables(expr)
            var = vars_found[0] if vars_found else sp.Symbol('x')
            result = sp.diff(expr, var)
            result_simplified = sp.simplify(result)
            sympy_solution_str = f"d/d{var}({expr}) = {result_simplified}"
            steps.append(f"**Differentiating:** `{expr}` w.r.t. `{var}`")
            steps.append(f"**Result:** `{result_simplified}` ✅")
            solution_value = [result_simplified]
        except Exception as e:
            steps.append(f"**Error:** {e}")

    # ─── INTEGRAL ────────────────────────────────────────────────────
    elif any(k in tl for k in ['integral', 'integrate', '∫']):
        expr_str = _extract_sympy_expr(problem_text)
        try:
            expr = parse_expr(expr_str, transformations=TRANSFORMATIONS)
            vars_found = _detect_variables(expr)
            var = vars_found[0] if vars_found else sp.Symbol('x')
            result = sp.integrate(expr, var)
            sympy_solution_str = f"∫({expr})d{var} = {result} + C"
            steps.append(f"**Integrating:** `{expr}` w.r.t. `{var}`")
            steps.append(f"**Final Answer:** `{sympy_solution_str}` ✅")
            solution_value = [result]
        except Exception as e:
            steps.append(f"**Error:** {e}")

    # ─── SYSTEMS / EQUATIONS ─────────────────────────────────────────
    elif '=' in problem_text:
        cleaned = _strip_prefix(problem_text)
        # Check for multiple equations (separated by comma or semicolon or "and")
        eqs_raw = re.split(r',|;|and', cleaned)
        eqs_to_solve = []
        try:
            for raw_eq in eqs_raw:
                if '=' in raw_eq:
                    lhs_str, rhs_str = raw_eq.split('=', 1)
                    lhs = parse_expr(lhs_str.strip(), transformations=TRANSFORMATIONS)
                    rhs = parse_expr(rhs_str.strip(), transformations=TRANSFORMATIONS)
                    eqs_to_solve.append(sp.Eq(lhs, rhs))
            
            if eqs_to_solve:
                all_vars = set()
                for eq in eqs_to_solve:
                    all_vars.update(eq.free_symbols)
                all_vars = sorted(list(all_vars), key=lambda s: str(s))
                
                steps.append(f"**System detected:** {', '.join([str(e) for e in eqs_to_solve])}")
                solutions = sp.solve(eqs_to_solve, all_vars)
                
                if solutions:
                    solution_value = [solutions]
                    sympy_solution_str = str(solutions)
                    steps.append(f"**Solution:** `{solutions}` ✅")
                else:
                    sympy_solution_str = "No solution found."
                    steps.append("**Result:** No solutions.")
        except Exception as e:
            steps.append(f"**Error solving equation:** {e}")

    # ─── SIMPLIFY ────────────────────────────────────────────────────
    else:
        expr_str = _normalize_nl(_strip_prefix(problem_text))
        try:
            expr = parse_expr(expr_str, transformations=TRANSFORMATIONS)
            simplified = sp.simplify(expr)
            
            # If nothing changed and no math operations were present, likely not a problem
            if str(simplified) == expr_str and not any(c in expr_str for c in '+-*/^()'):
                sympy_solution_str = f"'{problem_text}' is not a clear math problem."
                steps.append(f"**Note:** Input is likely not a math problem.")
            else:
                sympy_solution_str = f"= {simplified}"
                steps.append(f"**Expression:** `{expr}`")
                steps.append(f"**Simplified:** `{simplified}` ✅")
                solution_value = [simplified]
        except Exception as e:
            steps.append(f"**Parse error:** {e}")


    # RAG context
    if retrieved_context:
        steps.append(f"\n**Context Used:**\n> {retrieved_context[:200]}...")

    return {
        "solution": sympy_solution_str or "Could not solve symbolically.",
        "steps": steps,
        "sympy_verified": solution_value is not None,
        "solution_value": str(solution_value[0]) if solution_value else None,
    }
