import numpy as np
from ._base_measurement import BaseMeasurement


class MeasurementOptical(BaseMeasurement):
    """Angle (line-of-sight) and angle-rate measurement object relative to an observer"""

    def __init__(self):
        super().__init__()
        self.name = "Optical"
        self.measurement_dim = 4
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
        S = np.eye(2, 3)
        
        x_observer, sigma_p, dt = params

        rho = x[:3] - x_observer[:3]
        eta = x[3:6] - x_observer[3:6]
        Tc = _get_transformation_matrix(rho)

        rho_c = Tc @ rho
        eta_c = Tc @ eta

        p = 1 / rho_c[2] * S @ rho_c
        pdot = 1 / rho_c[2] * S @ (eta_c - (eta_c[2]/rho_c[2]) * rho_c)

        y = np.concatenate((p, pdot)) + sigma_p * np.random.randn(4) # z = h(x) + v, v ~ N(0,R)

        R = sigma_p**2 * np.block([[np.eye(2), np.zeros(shape=(2,2))], [np.zeros(shape=(2,2)), (2/(dt**2))*np.eye(2)]])
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

        S = np.eye(2, 3)

        x_observer, _, _ = params
        assert len(x_observer) == 6, "Observer position must contain both position and velocity"
        assert len(x) == 6, "State vector must contain both position and velocity"
        rho = x[:3] - x_observer[:3]
        eta = x[3:6] - x_observer[3:6]
        Tc = _get_transformation_matrix(rho)

        rho_c = Tc @ rho
        eta_c = Tc @ eta

        p = 1 / rho_c[2] * S @ rho_c
        pdot = 1 / rho_c[2] * S @ (eta_c - (eta_c[2]/rho_c[2]) * rho_c)

        return np.concatenate((p, pdot))
    
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
            (np.ndarray): 4-by-6 measurement partials matrix
        """
        S = np.eye(2, 3)
        e3  = np.array([0, 0, 1])   
        
        x_observer, _, _ = params
        rho = x[:3] - x_observer[:3]
        nu = x[3:6] - x_observer[3:6]

        T_c = self._get_transformation_matrix(rho)
        rho_c = T_c @ rho
        nu_c = T_c @ nu

        A = (S / rho_c[2]) @ (np.eye(3) - np.outer(rho_c, e3) / rho_c[2]) @ T_c
        B = (-S / (rho_c[2]**2)) @ (nu_c[2] * np.eye(3) + np.outer(nu_c, e3) - 2 * (nu_c[2] / rho_c[2]) * np.outer(rho_c, e3)) @ T_c

        H = np.block([[A, np.zeros(shape=(2, 3))],
                      [B, A                     ]])

        return H
    
def _get_transformation_matrix(rho: np.ndarray[float], eps = 1e-2):

    v3 = rho / np.linalg.norm(rho)

    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    
    v1 = np.cross(e2, v3)
    if np.linalg.norm(v1) > eps:
        v1 /= np.linalg.norm(v1)
    else:
        v1 = np.cross(e1, v3)
        v1 /= np.linalg.norm(v1)

    v2 = np.cross(v3, v1)
    v2 /= np.linalg.norm(v2)
    
    T_c = np.vstack((v1, v2, v3))

    return T_c