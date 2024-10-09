"""Test for dynamics class"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import rudolfpy as rd

def test_dynamics_cr3bp():
    # initialize integrator
    mu = 1.215058560962404e-02
    integrator = rd.DynamicsCR3BP(mu = mu, method='DOP853', rtol = 1e-12, atol = 1e-12)

    # initial state, propagation time
    x0 = np.array([1.0809931218390707E+00,
          0.0000000000000000E+00,
          -2.0235953267405354E-01,
          1.0157158264396639E-14,
          -1.9895001215078018E-01,
          7.2218178975912707E-15])
    period = 2.3538670417546639E+00
    tspan = (0, period)

    # propagate (just the initial state)
    sol = integrator.solve(tspan, x0)
    assert np.allclose(sol.y[:,-1], x0, atol=1e-11),\
        f"Difference: {sol.y[:,-1] - x0}"

    # propagate (state & STM) -- we always just need to provide the initial state, but set stm = True
    sol_stm = integrator.solve(tspan, x0, stm=True)
    print(f"Final STM: {sol_stm.y[6:,-1].reshape(6,6)}")

    # propagate until we are at Earth-Moon rotating frame (where y == 0), as soon as we travelled for 0.25*period
    t_min_elapse = 0.25 * period
    def hit_EMframe(t, x, params):
        if t > t_min_elapse:
            return x[1]
        else:
            return np.nan
    hit_EMframe.terminal = True
    hit_EMframe.direction = 0
    sol_event = integrator.solve(tspan, x0, events=hit_EMframe)
    print(f"status (1 if event was detected): {sol_event.status}")
    assert sol_event.status == 1
    
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.y[0,:], sol.y[1,:], sol.y[2,:], lw=2.0)
    ax.plot(sol_event.y[0,:], sol_event.y[1,:], sol_event.y[2,:], lw=1.0)
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    return


if __name__=="__main__":
    test_dynamics_cr3bp()
    print("Done!")
    plt.show()