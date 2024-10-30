"""Angles-based measurement models"""

import numpy as np
from ._base_measurement import BaseMeasurement


class MeasurementAngle(BaseMeasurement):
    """Angle (line-of-sight) measurement object relative to an observer"""



    def __init__(self):
        super().__init__()
        self.measurement_dim = 3
        self.name = "Angle"
        return

    def func_simulate_measurements(self,
                                   t: float,
                                   x: np.ndarray[float],
                                   params: list):

        """Simulate angle measurement with noise
        
        Args:
            t (float): time
            x (np.ndarray[float]): state vector
            params (list): observer position vector and measurement noise standard deviation 
        
        Returns:
            (tuple): measurement and measurement covariance
        """
        r_observer, sigma_phi = params
        r_rel = x[:3] - r_observer
        rnorm = np.linalg.norm(r_rel)
        T_ptrb = get_perturbation_T(sigma_phi)
        y = T_ptrb @ r_rel/rnorm
        R = sigma_phi**2 * np.eye(3)
        return y, R
    
    def predict_measurement(self,
                            t: float,
                            x: np.ndarray[float], 
                            params: list):

        """Generate measurement prediction
        
        Args:
            t (float): time
            x (np.ndarray[float]): state vector
            params (list) : containing r_observer (np.ndarray) observer position vector
            

        Returns:
            (np.ndarray): measurement prediction
        """
        r_observer, _ = params

        assert len(r_observer) == 3, "Observer position must be 3D"
        assert len(x) >= 3, "State vector must contain the position (and optionally velocity)"
        r_rel = x[:3] - r_observer
        return r_rel/np.linalg.norm(r_rel)
    
    def measurement_partials(self,
                             t: float,
                             x: np.ndarray[float],
                             params:list):

        """Compute measurement partial derivatives
        
        Args:
            t (float): time
            x (np.ndarray): state vector
            params (list) : containing observer position vector
            
        Returns:
            (np.ndarray): 3-by-6 measurement partials matrix
        
        """
        r_observer, _ = params

        r_rel = x[:3] - r_observer
        rnorm = np.linalg.norm(r_rel)
        return np.concatenate((
            np.eye(3)/rnorm - np.outer(r_rel,r_rel)/rnorm**3,
            np.zeros((3,3)),
        ), axis=1)


class MeasurementAngleAngleRate(BaseMeasurement):
    """Angle (line-of-sight) and angle-rate measurement object relative to an observer"""



    def __init__(self):
        super().__init__()
        self.name = "Angle_AngleRate"
        self.measurement_dim = 6
        return

    def func_simulate_measurements(self,
                                   t: float,
                                   x: np.ndarray[float],
                                   params:list):

        """Simulate angle measurement with noise
        
        Args:
            t (float): time
            x (np.ndarray): state vector
            params (list): observer state, measurement noise standard deviation, and time step

        Returns:
            (tuple): measurement and measurement covariance
        """
        x_observer, sigma_phi, dt = params
        r_rel = x[:3] - x_observer[:3]
        v_rel = x[3:6] - x_observer[3:6]
        rnorm = np.linalg.norm(r_rel)
        T_ptrb = get_perturbation_T(sigma_phi)
        y = np.concatenate((
            T_ptrb @ r_rel/rnorm,
            T_ptrb @ (v_rel/rnorm - np.dot(r_rel, v_rel) * r_rel/rnorm**3)
        ))
        R = sigma_phi**2 * np.concatenate((
            np.concatenate((np.eye(3), np.zeros((3,3))), axis=1),
            np.concatenate((np.zeros((3,3)), 2/dt**2 * np.eye(3)), axis=1),
        ))
        return y, R

    def predict_measurement(self,
                            t: float,
                            x: np.ndarray[float],
                            params:list):

        """Generate measurement prediction
        
        Args:
            t (float): time
            x (np.ndarray): state vector
            params (list): observer state, measurement noise standard deviation, and time step
        
        Returns:
            (np.ndarray): measurement prediction
        """

        x_observer, _, _ = params
        assert len(x_observer) == 6, "Observer position must contain both position and velocity"
        assert len(x) == 6, "State vector must contain both position and velocity"
        r_rel = x[:3] - x_observer[:3]
        v_rel = x[3:6] - x_observer[3:6]
        rnorm = np.linalg.norm(r_rel)
        return np.concatenate((
            r_rel/rnorm,
            v_rel/rnorm - np.dot(r_rel, v_rel) * r_rel/rnorm**3
        ))
    
    def measurement_partials(self,
                             t: float,
                             x: np.ndarray[float],
                             params:list):

        """Compute measurement partial derivatives
        
        Args:
            t (float): time
            x (np.ndarray): state vector
            params (list): observer state, measurement noise standard deviation, and time step

        Returns:
            (np.ndarray): 3-by-6 measurement partials matrix
        """
        x_observer, _, _ = params
        r_rel = x[:3] - x_observer[:3]
        v_rel = x[3:6] - x_observer[3:6]
        rnorm = np.linalg.norm(r_rel)
        H11 = np.eye(3)/rnorm - np.outer(r_rel,r_rel)/rnorm**3
        H21 = -np.outer(v_rel,r_rel) / rnorm**3 \
            -(np.outer(r_rel,v_rel) + np.dot(r_rel,v_rel)*np.eye(3)) / rnorm**3\
            + 3*np.dot(r_rel,v_rel) * np.outer(r_rel,r_rel) / rnorm**5
        return np.concatenate((
            np.concatenate((H11, np.zeros((3,3))), axis=1),
            np.concatenate((H21, H11), axis=1),
        ))
    

def vec2skewsymmetric(u):
    """Get 3-by-3 skew-symmetric matrix representation from vector `u`"""
    return np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])


def get_perturbation_T(dphi: float):
    """Get transformation matrix corresponding to small angle perturbation
    
    Args:
        dphi (float): angle perturbation, in radians
    
    Returns:
        (np.ndarray): perturbation matrix
    """
    # create random unit vector
    uvec = 2*np.random.rand(3) - 1
    uvec /= np.linalg.norm(uvec)

    # create random angle
    eps = np.random.normal(0, dphi)

    # compute transformation matrix via Rodrigues' rotation formula
    return np.cos(eps)*np.eye(3) + np.sin(eps)*vec2skewsymmetric(uvec) + (1 - np.cos(eps))*np.outer(uvec, uvec)
       




