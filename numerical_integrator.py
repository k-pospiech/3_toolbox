def trapezoidal_rule(x, y):
    n = len(x)
    integral = 0
    for i in range(1, n):
        integral += 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    return integral

def simpsons_rule(x, y):
    n = len(x) - 1
    if n % 2 == 1:
        raise ValueError("Number of intervals must be even for Simpson's rule.")
    
    integral = y[0] + y[-1]
    for i in range(1, n, 2):
        integral += 4 * y[i]
    for i in range(2, n-1, 2):
        integral += 2 * y[i]
    
    integral *= (x[1] - x[0]) / 3
    return integral

def rectangle_method(x, y):
    n = len(x)
    integral = 0
    for i in range(1, n):
        integral += y[i-1] * (x[i] - x[i-1])
    return integral

if __name__ == "__main__":
    x_values = [0, 1, 2, 3, 4]
    y_values = [0, 1, 4, 9, 16]
    
    print("Trapezoidal Rule:", trapezoidal_rule(x_values, y_values))
    print("Simpson's Rule:", simpsons_rule(x_values, y_values))
    print("Rectangle Method:", rectangle_method(x_values, y_values))