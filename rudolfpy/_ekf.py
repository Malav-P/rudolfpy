"""EKF object"""

import numpy as np

from ._base_filter import BaseFilter

class ExtendedKalmanFilter(BaseFilter):
    """Extended Kalman Filter object
    
    Args:
        dynamics (Dynamics): dynamics object
        measurement_model (Measurement): measurement object
        func_process_noise (func): function for process noisem, with signature `Q = func(tspan, x, params_Q)`
        params_Q (list): list of parameters for process noise function
    """
    def __init__(
        self,
        dynamics,
        measurement_model,
        func_process_noise,
        params_Q = [1e-5]
    ):
        super().__init__(
            dynamics = dynamics,
            measurement_model = measurement_model,
            func_process_noise = func_process_noise,
            params_Q = params_Q
        )
        self.name = "ExtendedKalmanFilter"

        return

    def summary(self):
        print(f" ************** {self.name} summary ************** ")
        print(f"   Dynamics model : {self.dynamics.rhs.__name__}")
        print(f"   Process noise model : {self.func_process_noise.__name__}")
        print(f"   Measurement model : {self.measurement_model.name}")
        return

    def initialize(self, t, x0, P0):
        self._t = t
        self._x = x0
        self._P = P0
    
    def predict(self, tspan):
        """Perform time prediction
        
        Args:
            tspan (tuple): time span for prediction
        
        Returns:
            x, P (tuple) : tuple of ndarray giving the state, covariance pair
        """
        # perform prediction of state
        sol_stm = self.dynamics.solve(tspan, self._x, stm=True)
        self._x = sol_stm.y[:6,-1]                                    # propagate state
        Phi = sol_stm.y[6:, -1].reshape(6,6)
        Q = self.func_process_noise(tspan, self.x, self.params_Q)     # process noise
        self._P = Phi @ self._P @ Phi.T + Q                           # propagate covariance
        self._t += tspan[1] - tspan[0]                                # propagate time

        return self._x, self._P
    
    def update(self, y, R, params = None):
        """Perform measurement update
        
        Args:
            y (np.ndarray): measurement vector
            R (np.ndarray): measurement covariance matrix

        Returns:
            x, P (tuple) : tuple of ndarray giving the state, covariance pair
        """
        m = len(y)
        assert R.shape == (m, m), f"R must be of shape ({m},{m})"
        # prediction of measurement and measurement partials
        if params is None:
            h = self.measurement_model.predict_measurement(self._t, self._x)
            H = self.measurement_model.measurement_partials(self._t, self._x)
        else:
            h = self.measurement_model.predict_measurement(self._t, self._x, params)
            H = self.measurement_model.measurement_partials(self._t, self._x, params)
            
        # perform measurement update
        ytilde = y - h                                                                 # innovation
        S = H @ self._P @ H.T + R                                                      # innovation covariance
        K = self._P @ H.T @ np.linalg.inv(S)                                           # Kalman gain
        self._x = self._x + K @ ytilde                                                 # update state estimate
        self._P = (np.eye(6) - K @ H) @ self._P @ (np.eye(6) - K @ H).T + K @ R @ K.T  # Joseph update
        
        return self._x, self._P