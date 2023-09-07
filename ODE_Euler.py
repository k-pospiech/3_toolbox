# Setting up the Blueprint (Base Class)
# Needs: time step ('dt'), 
# methods: 'step' to advance the solution one step at the time, 
# and 'solve' to solve the equation over given time range

class NumericalMethod:
    """
    A base class for numerical integration methods.
    
    Attributes:
        dt (float): Time step for integration.
        
    Methods:
        step(y, t): Placeholder method for advancing the solution by one step.
        solve(y0, t0, t_end): Solves the equation over a given time range and returns a history of (t, y) values.
    """
    def __init__(self,dt):
        """
        Initializes the NumericalMethod with a specified time step.
        
        Args:
            dt (float): Time step for the integration method.
        """
        self.dt = dt

    def step(self, y, t):
        """Placeholder method for advancing the solution by one step.""" 
        pass

    def solve(self, y0, t0, t_end):
        """
        Solves the differential equation over a given time range.
        
        Args:
            y0 (float): Initial value of the solution.
            t0 (float): Start time.
            t_end (float): End time.
            
        Returns:
            list: A list of tuples containing time (t) and solution (y) pairs.
        """
        y = y0
        t = t0
        history = [(t,y)]

        while t < t_end:
            y = self.step(y,t)
            t += self.dt
            history.append((t,y))

        return history

# Creating a subclass that uses the structure of NumericalMethod but implements specifics of Euler method

class EulerMethod(NumericalMethod):
    """
    Euler's method for solving differential equations, derived from NumericalMethod class.
    
    Attributes:
        dt (float): Time step for integration, inherited from NumericalMethod.
        derivative (function): The derivative function dy/dt of the differential equation.
        
    Methods:
        step(y, t): Advances the solution by one step using Euler's method.
    """
    def __init__(self, dt, derivative):
        """
        Initializes the EulerMethod with a specified time step and derivative function.
        
        Args:
            dt (float): Time step for the integration method.
            derivative (function): The derivative function dy/dt of the differential equation.
        """
        super().__init__(dt)
        self.derivative = derivative

    def step(self, y, t):
        """
        Advances the solution by one step using Euler's method.
        
        Args:
            y (float): Current solution value.
            t (float): Current time.
            
        Returns:
            float: Updated solution value after taking one step.
        """
        return y + self.dt * self.derivative(y, t)
    

# Example usage
# Exponential decay ODE: dy/dt = -ky

# def exponential_decay(y, t, k=1):
#     return -k*y

# # Solve with Euler method
# solver = EulerMethod(0.1, exponential_decay)
# history = solver.solve(1, 0, 5) # Starting y is 1, from t=0 to t=5

# for t, y in history:
#     print(f"t = {t:.2f}, y(t) = {y:.2f}")
