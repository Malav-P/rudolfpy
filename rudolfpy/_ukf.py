""" UKF Object"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional
from ._base_filter import BaseFilter

class UnscentedKalmanFilter(BaseFilter):
    
    def __init__(self,
                dynamics,
                measurement_model,
                func_process_noise,
                params_Q):

        super().__init__(
            dynamics = dynamics,
            measurement_model = measurement_model,
            func_process_noise = func_process_noise,
            params_Q = params_Q
        )

        self.name = "Unscented Kalman Filter"
        self.dim_z = self.measurement_model.measurement_dim


    def initialize(self,
                   t: float,
                   x0: np.ndarray[float],
                   P0: np.ndarray[float],
                   alpha: float = 1e-3,
                   beta: float = 2,
                   kappa: float = 0):
        """ 
        Initialize the UKF
        
        Args:
            t (float) : time
            x0 (np.ndarray[float]) : initial state estimate
            P0 (np.ndarray[float]) : initial covariance
            alpha (float) : alpha parameter
            beta (float) : beta paramter
            kappa (float) : kappa parameter
        
        """
        self._t = t
        self._x = x0
        self._P = P0

        self.dim_x = x0.size
        self.n_sigma = 2 * self.dim_x + 1

        # scaling parameters
        self.lamda = (alpha**2) * (self.dim_x + kappa) - self.dim_x

        # unscented weights
        self.W0m = self.lamda / (self.dim_x + self.lamda)
        self.Wim = 0.5 / (self.dim_x + self.lamda)
        self.W0c = self.lamda / (self.dim_x + self.lamda) + (1 - alpha**2 + beta)
        self.Wic = self.Wim

        self.Wm = np.hstack(([self.W0m], self.Wim * np.ones(self.n_sigma - 1) )) 
        self.Wc = np.hstack(([self.W0c], self.Wic * np.ones(self.n_sigma - 1) )) 

    
    

    def predict(self,
                tspan: ArrayLike):
        """
        Predict step

        Args:
            tspan (ArrayLike): 2-tuple containing timespan of predict step

        Returns:
            x, P : new state and covariance estimate
        
        """

        # compute sigma points
        sigma_pts = self.sigma_points() 

        # propagate sigma points
        y_sigmas = np.zeros(shape=(self.n_sigma, self.dim_x))

        for i in range(self.n_sigma):
            y_sigmas[i] = self.f(sigma_pts[i], tspan=tspan)

        y, Pyy = self.compute_mean_and_covariances(y_sigmas)

        # add the process noise 
        Q = self.func_process_noise(tspan, y, self.params_Q)
        Pyy += Q

        self._x = y
        self._P = Pyy
        self._t += tspan[1] - tspan[0]

        return y, Pyy

    def update(self,
               z_measured: np.ndarray[float],
               R: np.ndarray[float],
               params: Optional[list] = None):
        """
        Update step

        Args:
            z_measured (np.ndarray[float]) : measurement vector
            R (np.ndarray[float]) : measurement covariance
            params (list) : parameters needed for measurement prediction

        Returns:
            x, P : posterior state and covariance estimate
        """

        # compute sigma pts
        sigma_pts = self.sigma_points()

        # propagate through measurement model
        z_sigmas = np.zeros(shape=(self.n_sigma, self.dim_z))

        for i in range(self.n_sigma):
            z_sigmas[i] = self.h(sigma_pts[i], params = params)

        z, Pzz = self.compute_mean_and_covariances(z_sigmas)

        # add measurement noise
        Pzz += R

        # weighted centered cross correlation
        Pyz = (sigma_pts - self._x).T @ np.diag(self.Wc) @ (z_sigmas - z)

        # gain matrix
        K = Pyz @ np.linalg.inv(Pzz)

        # state and cov update
        self._x += K @ (z_measured - z)
        self._P -= K @ Pzz @ K.T

        return self._x, self._P



    def compute_mean_and_covariances(self,
                                     y_sigmas: np.ndarray[float]):
        """
        Compute weighted mean and weighted covariance of 2d array

        Args:
            y_sigmas (np.ndarray[float]) : array of shape (n_points, dim)

        Returns:
            y, Pyy : weighted mean and covariance
        
        """
        
        y = np.average(y_sigmas, axis=0, weights=self.Wm)

        Pyy = (y_sigmas - y).T @ np.diag(self.Wc) @ (y_sigmas - y)

        return y, Pyy


    def h(self,
          x: np.ndarray[float],
          params: Optional[list] = None):

        """
        Measurement model functon. Represent z = h(x) where z is measurement and x is state.

        Args:
            x (np.ndarray[float]) : state vector
            params (list) : list of parameters to pass to measurement model
        """

        if params is None:
            z = self.measurement_model.predict_measurement(self._t, x)  # measurement_prediction
        else:
            z = self.measurement_model.predict_measurement(self._t, x, params)  # measurement_prediction

        return z


    def f(self,
          x: np.ndarray[float],
          tspan: ArrayLike):

        """
        Dynamics model function. Represent x_t+1 = f(x_t, t)

        Args:
            x (np.ndarray[float]) : state vector
            tspan (ArrayLike) : 2-tuple representing time span of dynamics propagation
        """

        sol_stm = self.dynamics.solve(tspan, x, stm=False)
        y = sol_stm.y[:self.dim_x, -1]

        return y



    def sigma_points(self):
        """
        Computes sigma points from state estimate

        Returns:
            sigma_pts (np.ndarray[float]): 2d array of shape (n_sigma_points, state_dimension)
        """

        sigma_pts = np.zeros(shape=(self.n_sigma, self.dim_x))

        S = np.linalg.cholesky(self._P).T

        sigma_pts[0] = self._x
        plusterm = np.sqrt(self.dim_x + self.lamda) * S + self._x
        minusterm = -np.sqrt(self.dim_x + self.lamda) * S + self._x

        sigma_pts[1:self.dim_x + 1] = plusterm
        sigma_pts[self.dim_x + 1:] = minusterm
        
        
        return sigma_pts