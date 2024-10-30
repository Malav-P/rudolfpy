"""Position measurement object"""

import numpy as np
from ._base_measurement import BaseMeasurement

class MeasurementPosition(BaseMeasurement):
    """Position vector measurement object"""



    def __init__(self):
        super().__init__()
        self.name = "PositionVector"
        self.measurement_dim = 3
        return
    
    def predict_measurement(self,
                            t: float,
                            x: np.ndarray[float],
                            params:list):
        """
        Compute measurement prediction

        Args:
            t (float): time
            x (np.ndarray): state vector
            params (list): list with one element, the standard deviation of the measurement noise

        Returns:
            (np.ndarray) : predicted measurement
        """
        return x[:3]
    
    def measurement_partials(self,
                             t: float,
                             x: np.ndarray[float],
                             params:list):
        """
        Compute measurement jacobian

        Args:
            t (float): time
            x (np.ndarray): state vector
            params (list): list with one element, the standard deviation of the measurement noise

        Returns:
            (np.ndarray) : observation jacobian
        
        """

        return np.concatenate((np.eye(3), np.zeros((3,3))), axis=1)
    

    def func_simulate_measurements(self,
                                   t: float,
                                   x: np.ndarray[float],
                                   params:list):

        """Simulate position vector measurements with noise
        
        Args:
            t (float): time
            x (np.ndarray): state vector
            params (list): list with one element, the standard deviation of the measurement noise

        Returns:
            (tuple): measurement and measurement covariance
        """
        sigma_r = params[0]
        y = x[0:3] + np.random.normal(0, sigma_r, 3)
        R = sigma_r**2 * np.eye(3)
        return y, R