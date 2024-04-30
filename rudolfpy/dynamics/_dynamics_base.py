"""Object and constructor for dynamics"""

import numpy as np

class BaseDynamics:
    """Base dynamics class to be inherited by specific dynamics for the filter"""
    def __init__(self, propagator_function):
        self.name = "DynamicsBase"
        self.propagator_function = propagator_function
        return
    
    def summary(self):
        return
    
    def solve(self, tspan, x0, teval=None, events=None, stm=False, stm0=None):
        raise NotImplementedError
    
    def get_xdot(self, t, x):
        raise NotImplementedError