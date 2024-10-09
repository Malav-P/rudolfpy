"""Test EKF implementation"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import rudolfpy as rd

np.random.seed(100)

def test_ekf_recurse_event():
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
    print(f"Period: {period*TU/86400:1.4f} days")

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

    def events_placeholder(t,y,p=None):
        return np.dot(y[0:3], y[3:6])
    events_placeholder.terminal = True
    events_placeholder.direction = -1     # going from neg --> pos radial velocity (i.e. apolune)

    # perform recursion
    tspan = [0.0, 5.5 * period]
    meas_interval_min = 0.75*period
    print(f"tspan[1] = {tspan[1]:1.4f} TU")
    recursor.recurse_measurements_func_events(
        tspan,
        x0,
        x0hat,
        P0,
        events = [events_placeholder,],
        func_simulate_measurements = rd.func_simulate_measurements,
        params_measurement_constant = (sigma_r,),
        meas_interval_min = meas_interval_min,
    )
    print(f"Number of measurements: {len(recursor.ys)}")

    # plot recursion results
    fig, axs = recursor.plot_state_history(
        TU = TU/86400,
        time_unit = "day",
        state_multipliers = [LU,LU,LU,VU,VU,VU],
        state_labels = ["$x$, km","$y$, km","$z$, km",
                        "$v_x$, km/s","$v_y$, km/s","$v_z$, km/s"],
    )
    for ax in axs.flatten():
        for _t in recursor.ts_y:
            ax.axvline(_t*TU/86400, color='r', linestyle='--')

    fig, axs = recursor.plot_error_history(
        TU = TU/86400,
        time_unit = "day",
        state_multipliers = [LU,LU,LU,1e3*VU,1e3*VU,1e3*VU],
        state_labels = ["$delta x$, km","$delta y$, km","$delta z$, km",
                        "$delta v_x$, m/s","$delta v_y$, m/s","$delta v_z$, m/s"],
    )
    for ax in axs.flatten():
        for _t in recursor.ts_y:
            ax.axvline(_t*TU/86400, color='r', linestyle='--')

    fig, axs = recursor.plot_gain_history(TU = TU/86400, time_unit = "day")
    assert True


if __name__=="__main__":
    test_ekf_recurse_event()
    print("Done!")
    plt.show()