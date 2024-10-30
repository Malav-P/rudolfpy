"""Base filter object"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

def unbiased_random_process_3dof(tspan, x, params):
    """Unbiased random process for Newtonian motion in 3-DOF
    
    Args:
        tspan (tuple): time span
        x (np.array): state vector # TODO NOT USED, CAN BE REMOVED
        params (list): list containing sigma_process, which is a tunable parameter

    Returns:
        (np.array): process noise matrix (6x6)
    """
    dt_abs = np.abs(tspan[1] - tspan[0])
    return params[0]**2 * np.concatenate((
        np.concatenate((dt_abs**3/3*np.eye(3), dt_abs**2/2*np.eye(3)), axis=1),
        np.concatenate((dt_abs**2/2*np.eye(3), dt_abs*np.eye(3)), axis=1),
    ))

class BaseFilter(ABC):
    def __init__(
        self,
        dynamics,
        measurement_model,
        func_process_noise = unbiased_random_process_3dof,
        params_Q = [1e-5]
    ):
        self.dynamics = dynamics
        self.measurement_model = measurement_model
        self.func_process_noise = func_process_noise
        self.params_Q = params_Q

        self.name = "BaseFilter"

        # set initial time
        self._t = 0.0
        return

    @abstractmethod
    def initialize(self, t, x0, P0):
        """initialize filter with intial state and covariance"""
        pass

    @abstractmethod
    def predict(self, tspan) -> Tuple[np.ndarray, np.ndarray]:
        """predict step must be implemented by derived filter class"""

        """ return type must be (x, P) state, covariance matrix pair"""
        pass

    @abstractmethod
    def update(self, y, R, params) -> Tuple[np.ndarray, np.ndarray]:
        """update step must be implemented by derived filter class"""

        """ return type must be (x, P) state, covariance matrix pair"""
        pass

    @property
    def t(self):
        """getter for time"""
        return self._t
    
    @t.setter
    def t(self, value):
        """setter for time"""
        self._t = value
        return
    
    @property
    def x(self):
        """getter for state estimate vector"""
        return self._x
    
    @x.setter
    def x(self, value):
        """setter for state estimate vector"""
        self._x = value
        return
    
    @property
    def P(self):
        """getter for state covariance matrix"""
        return self._P
    
    @P.setter
    def P(self, value):
        """setter for state covariance matrix"""
        self._P = value
        return