"""Angles-based measurement models"""

import numpy as np
from ._base_measurement import BaseMeasurement


class MeasurementAngle(BaseMeasurement):
    """Angle (line-of-sight) measurement object"""
    def __init__(self):
        super().__init__()
        return
    
    def predict_measurement(self, t, x, r_observer):
        r_rel = x[0:3] - r_observer
        return r_rel/np.linalg.norm(r_rel)
    
    def measurement_partials(self, t, x, r_observer):
        r_rel = x[0:3] - r_observer
        rnorm = np.linalg.norm(r_rel)
        return np.concatenate((
            np.eye(3)/rnorm - np.outer(r_rel,r_rel)/rnorm**3,
            np.zeros((3,3)),
        ), axis=1)
    

def func_simulate_measurement_angle(t, x, r_observer, sigma_exp):
    """Simulate angle measurement with noise
    
    Args:
        x (np.ndarray): state vector
        r_observer (np.ndarray): observer position vector
        sigma_exp (float): measurement noise standard deviation
    
    Returns:
        (tuple): measurement and measurement covariance
    """
    r_rel = x[0:3] - r_observer
    rnorm = np.linalg.norm(r_rel)
    y = r_rel/rnorm + np.random.normal(0, sigma_exp, 3)
    y /= np.linalg.norm(y)         # normalize line of sight with noise
    R = sigma_exp**2 * np.eye(3)
    return y, R