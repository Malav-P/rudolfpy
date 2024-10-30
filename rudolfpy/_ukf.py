""" UKF Object"""

import numpy as np

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


    def initialize(self, t, x0, P0):
        self._t = t
        self._x = x0
        self._P = P0

        self.dim_x = x0.size
        self.n_sigma = 2 * self.dim_x + 1

        # scaling parameters
        alpha = 1e-3
        beta = 2
        kappa = 0
        self.lamda = (alpha**2) * (self.dim_x + kappa) - self.dim_x

        # unscented weights
        self.W0m = self.lamda / (self.dim_x + self.lamda)
        self.Wim = 0.5 / (self.dim_x + self.lamda)
        self.W0c = self.lamda / (self.dim_x + self.lamda) + (1 - alpha**2 + beta)
        self.Wic = self.Wim

        self.Wm = np.hstack(([self.W0m], self.Wim * np.ones(self.n_sigma - 1) )) 
        self.Wc = np.hstack(([self.W0c], self.Wic * np.ones(self.n_sigma - 1) )) 

    
    

    def predict(self, tspan):

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

    def update(self, z_measured, R, params = None):

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



    def compute_mean_and_covariances(self, y_sigmas):
        
        y = np.average(y_sigmas, axis=0, weights=self.Wm)

        Pyy = (y_sigmas - y).T @ np.diag(self.Wc) @ (y_sigmas - y)

        return y, Pyy


    def h(self, sigma_pt, params = None):

        if params is None:
            z = self.measurement_model.predict_measurement(self._t, sigma_pt)  # measurement_prediction
        else:
            z = self.measurement_model.predict_measurement(self._t, sigma_pt, params)  # measurement_prediction

        return z


    def f(self, sigma_pt, tspan):
        sol_stm = self.dynamics.solve(tspan, sigma_pt, stm=False)
        y_sigma = sol_stm.y[:self.dim_x, -1]

        return y_sigma



    def sigma_points(self):

        sigma_pts = np.zeros(shape=(self.n_sigma, self.dim_x))

        sigma_pts[0] = self._x
        plusterm = np.sqrt(self.dim_x + self.lamda) * np.linalg.cholesky(self._P).T + self._x
        minusterm = -np.sqrt(self.dim_x + self.lamda) * np.linalg.cholesky(self._P).T + self._x

        sigma_pts[1:self.dim_x + 1] = plusterm
        sigma_pts[self.dim_x + 1:] = minusterm
        
        
        return sigma_pts


