"""Test EKF implementation"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import rudolfpy as rd

np.random.seed(100)

def test_ekf_cr3bp():
    # initialize integrator
    mu = 1.215058560962404e-02
    dynamics = rd.DynamicsCR3BP(mu = mu, method='DOP853', rtol = 1e-12, atol = 1e-12)

    # initial state, propagation time
    x0 = np.array([1.0809931218390707E+00,
          0.0000000000000000E+00,
          -2.0235953267405354E-01,
          1.0157158264396639E-14,
          -1.9895001215078018E-01,
          7.2218178975912707E-15])
    period = 2.3538670417546639E+00
    tspan = (0, period)

    # initialize position measurements object
    meas_model = rd.MeasurementPosition()

    # initialize EKF
    params_Q = [1e-5,]
    filter = rd.ExtendedKalmanFilter(dynamics, meas_model,
                                     func_process_noise = rd.unbiased_random_process_3dof,
                                     params_Q = params_Q,)
    filter.summary()
    
    x0hat = x0 + np.random.normal(0, 1e-6, 6)
    P0 = np.eye(6) * 1e-6

    filter.t = 0.0              # set initial time
    filter.nx = len(x0hat)      # set number of states
    filter.x = x0hat            # set initial state estimate
    filter.P = P0               # set initial state covariance matrix
    print(f"Initial filter state: ")
    print(f"filter.x = {filter.x}")
    print(f"filter.P = {filter.P}")

    # perform prediction
    tspan = [0.0, 0.5]
    filter.predict(tspan)
    print(f"After prediction over time span {tspan}: ")
    print(f"filter.x = {filter.x}")
    print(f"filter.P = {filter.P}")

    # perform measurement update
    sigma_r = 1e-3
    ymeas = filter.x[0:3] + np.random.normal(0, sigma_r, 3)
    R = sigma_r**2 * np.eye(3)
    filter.update(ymeas, R)
    print(f"After measurement update: ")
    print(f"filter.x = {filter.x}")
    print(f"filter.P = {filter.P}")
    return


if __name__=="__main__":
    test_ekf_cr3bp()
    print("Done!")