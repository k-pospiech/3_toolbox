def absolute_error(X, Y):
    """
    Calculate the absolute error between two values.
    
    Args:
        X (float): First value, typically the value from simulation or estimation.
        Y (float): Second value, typically the reference or true value.
        
    Returns:
        float: The absolute difference between X and Y.
    """
    return abs(X - Y)

def relative_error(X, Y):
    """
    Calculate the relative error between two values.
    
    Args:
        X (float): First value, typically the value from simulation or estimation.
        Y (float): Second value, typically the reference or true value.
        
    Returns:
        float: The relative error, calculated as the absolute error divided by the absolute value of Y.
    """
    return abs(X - Y) / abs(Y)

def percentage_error(X, Y):
    """
    Calculate the percentage error between two values.
    
    Args:
        X (float): First value, typically the value from simulation or estimation.
        Y (float): Second value, typically the reference or true value.
        
    Returns:
        float: The percentage error, calculated as the relative error multiplied by 100.
    """
    return relative_error(X, Y) * 100

# # Example usage:
# X = 105  # simulation result
# Y = 100  # experimental result

# print("Absolute Error:", absolute_error(X, Y))
# print("Relative Error:", relative_error(X, Y))
# print("Percentage Error:", percentage_error(X, Y))