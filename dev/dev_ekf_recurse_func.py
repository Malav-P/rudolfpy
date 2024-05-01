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
    LU = 389703
    TU = 382981
    VU = LU / TU
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
    params_Q = [1e-6,]
    filter = rd.ExtendedKalmanFilter(dynamics, meas_model,
                                     func_process_noise = rd.unbiased_random_process_3dof,
                                     params_Q = params_Q,)
    filter.summary()
    
    # initial state estimate
    sigma_r0 = 100 / LU
    sigma_v0 = 0.001 / VU
    x0hat = x0 + np.array([sigma_r0]*3 + [sigma_v0]*3) * np.random.normal(0, 1, 6)
    P0 = np.diag([sigma_r0]*3 + [sigma_v0]*3)**2

    # create recursion object
    recursor = rd.Recursor(filter)

    # measurement frequency and simulation function
    sigma_r = 100 / LU
    t_measurements = np.linspace(0.05, 3 * period, 6)
    def func_simulate_measurements(t,x):
        y = x[0:3] + np.random.normal(0, sigma_r, 3)
        R = sigma_r**2 * np.eye(3)
        return y, R

    # perform recursion
    recursor.recurse_measurements_func(
        [0.0, t_measurements[-1]],
        x0,
        x0hat,
        P0,
        t_measurements,
        func_simulate_measurements
    )

    # plot recursion results
    fig, axs = recursor.plot_state_history(
        TU = TU/86400,
        time_unit = "day",
        state_multipliers = [LU,LU,LU,VU,VU,VU],
        state_labels = ["$x$, km","$y$, km","$z$, km",
                        "$v_x$, km/s","$v_y$, km/s","$v_z$, km/s"],
    )

    fig, axs = recursor.plot_error_history(
        TU = TU/86400,
        time_unit = "day",
        state_multipliers = [LU,LU,LU,1e3*VU,1e3*VU,1e3*VU],
        state_labels = ["$\delta x$, km","$\delta y$, km","$\delta z$, km",
                        "$\delta v_x$, m/s","$\delta v_y$, m/s","$\delta v_z$, m/s"],
    )
    return


if __name__=="__main__":
    test_ekf_cr3bp()
    print("Done!")
    plt.show()