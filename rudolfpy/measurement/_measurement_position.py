"""Position measurement object"""

import numpy as np
from ._base_measurement import BaseMeasurement

class MeasurementPosition(BaseMeasurement):
    """Position vector measurement object"""
    def __init__(self):
        super().__init__()
        return
    
    def predict_measurement(self, t, x):
        return x[0:3]
    
    def measurement_partials(self, t, x):
        return np.concatenate((np.eye(3), np.zeros((3,3))), axis=1)
    

def func_simulate_measurements(t, x, sigma_r):
    """Simulate position vector measurements with noise
    
    Args:
        x (np.ndarray): state vector
        sigma_r (float): standard deviation in position unit

    Returns:
        (tuple): measurement and measurement covariance
    """
    y = x[0:3] + np.random.normal(0, sigma_r, 3)
    R = sigma_r**2 * np.eye(3)
    return y, R