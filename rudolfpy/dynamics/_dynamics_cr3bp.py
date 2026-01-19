"""
Integrate CR3BP module
"""

import numpy as np
from scipy.integrate import solve_ivp

from ._dynamics_base import BaseDynamics
from ._eom_cr3bp import rhs_cr3bp, rhs_cr3bp_with_stm


class DynamicsCR3BP(BaseDynamics):
    """CR3BP dynamics class"""
    def __init__(self, mu, LU, TU, method="DOP853", rtol=1.e-13, atol=1.e-13):
        """Constructor
        
        Args:
            mu (float): cr3bp parameter
            method (str): integration method
            rtol (float): relative tolerance
            atol (float): absolute tolerance
        """
        super().__init__(None, LU, TU)
        self.mu = mu
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self.rhs = rhs_cr3bp
        self.rhs_stm = rhs_cr3bp_with_stm
        return

    def solve(self, tspan, x0, t_eval=None, events=None, stm=False, stm0=None):
        """Propagate CR3BP state

        Args:
            tspan (tuple): time-span `(t0,tf)`
            x0 (np.array): initial state-vector
            t_eval (np.array or None): time-steps to obtain solution
            event (callable): event functions, following `scipy.integrate.solve_ivp` syntax requirements
            stm (bool): whether to proppagate STM

        Returns:
            (bunch object): returned object from `scipy.integrate.solve_ivp`
        """
        assert len(x0) == 6

        # check if STM is required
        if stm is False:
            # run solve_ivp
            sol = solve_ivp(
                fun=self.rhs,
                t_span=tspan,
                y0=x0,
                events=events,
                t_eval=t_eval,
                args=(self.mu,),
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
            )
        else:
            # extend state-vector
            if stm0 is None:
                x0ext = np.concatenate((x0,np.reshape(np.eye(6),(36,))))
            else:
                assert stm0.shape == (6,6), "stm0 must be a 6x6 matrix"
                x0ext = np.concatenate((x0,stm0.reshape(6,6)))
            # run solve_ivp
            sol = solve_ivp(
                fun=self.rhs_stm,
                t_span=tspan,
                y0=x0ext,
                events=events,
                t_eval=t_eval,
                args=(self.mu,),
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
            )
        return sol
    
    def get_xdot(self, t, x):
        """Get state-derivative
        
        Args:
            t (float): time
            x (np.array): state-vector
        
        Returns:
            (np.array): state-derivative
        """
        return self.rhs(t, x, self.mu)