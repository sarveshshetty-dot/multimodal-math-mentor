import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class MathSolverTool:
    """Wrapper around SymPy to safely evaluate mathematical expressions and solve equations."""
    
    @staticmethod
    def solve_equation(equation_str: str) -> str:
        """
        Attempts to parse and solve an equation or simplify a mathematical expression.
        """
        try:
            transformations = (standard_transformations + (implicit_multiplication_application,))
            
            if "=" in equation_str:
                lhs_str, rhs_str = equation_str.split("=", 1)
                lhs = parse_expr(lhs_str, transformations=transformations)
                rhs = parse_expr(rhs_str, transformations=transformations)
                eq = sp.Eq(lhs, rhs)
                
                variables = list(eq.free_symbols)
                if not variables:
                    return str(eq == True) # Evaluates truthness if no vars
                
                solution = sp.solve(eq, variables[0])
                return f"Solutions for {variables[0]}: {solution}"
            else:
                expr = parse_expr(equation_str, transformations=transformations)
                simplified = sp.simplify(expr)
                return f"Simplified expression: {simplified}"
                
        except Exception as e:
            return f"Error using SymPy solver: {str(e)}"

    @staticmethod
    def calculate_derivative(expr_str: str, variable_str: str) -> str:
        """Calculates derivative of an expression with respect to a variable."""
        try:
            transformations = (standard_transformations + (implicit_multiplication_application,))
            expr = parse_expr(expr_str, transformations=transformations)
            var = sp.Symbol(variable_str)
            return str(sp.diff(expr, var))
        except Exception as e:
            return f"Error calculating derivative: {str(e)}"
