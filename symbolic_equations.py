from sympy import symbols
from sympy import simplify
from sympy import Eq
from sympy import solve
from sympy import S

def is_complex(val):
    """
    Check if a given value contains an imaginary unit.
    
    Parameters:
    - val: The value to be checked. It can be any sympy expression.
    
    Returns:
    - bool: True if the value contains an imaginary unit, otherwise False.
    """
    return val.has(S.ImaginaryUnit)

def pretty_output(variables, solutions):
    """
    Print solutions in a readable format.
    
    Parameters:
    - variables (list): A list of sympy symbols representing the unknowns.
    - solutions (list or dict): Solutions obtained from sympy's solve function. 
                                It can either be a list of values (for single equations) or 
                                a list of tuples (for systems of equations).
                                
    Returns:
    - None: This function only prints the results.
    """
    # If multiple solutions
    if isinstance(solutions, list) and all(isinstance(item,tuple) for item in solutions):
        for idx, solution_set in enumerate(solutions):
            formatted_solutions = [f"{var} = {sol}" for var, sol in zip(variables, solution_set)]
            print(f"Solution {idx + 1}: ")
            for formatted_sol in formatted_solutions:
                print(f"{formatted_sol}")
            print()
    else:
    # For single solutions
        for var, sol in zip(variables, solutions):
            print(f"{var} = {sol}")

# Defining symbols
x, y = symbols('x y')
# Defining domain
include_imaginary_solutions = False
domain_to_use = S.Complexes if include_imaginary_solutions else S.Reals

# Simplifying expressions
expr = x*x + 2*x*y + y*y
simplified_expr = simplify(expr)

# Solving single equations
equation = Eq(x + 2*y, 3)
solution = solve(equation, x, domain=domain_to_use)
if not solution:
    print("No solution found.")
else:
    pretty_output([x], solution)

# Solving systems of equations
eq1 = Eq(x + y, 3)
eq2 = Eq(2*x**4 + y, 4)
solutions = solve((eq1, eq2), (x,y), domain=domain_to_use)
solutions = [sol for sol in solutions if all(not is_complex(val) for val in sol) or include_imaginary_solutions]
if not solutions:
    print("No solutions found.")
else:
    pretty_output([x, y], solutions)