"""Object for recursing filter"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class Recursor:
    """Object for performing recursive simulation of filter.
    
    There are two main approaches to perform recursion:
    
    1. `recurse_measurements_func`: Recurse from a function that generates measurements
    2. `recurse_measurements_list`: Recurse from a given list of measurements
    """
    def __init__(
        self,
        filter,
    ):
        self.filter = filter 
        return
    
    def _initialize_filter(self, t0, x0_estim, P0):
        self.filter.t = t0         # set initial time
        self.filter.x = x0_estim   # set initial state estimate
        self.filter.P = P0         # set initial state covariance matrix
        self.filter.nx = len(x0_estim)      # number of states
        return
    
    def _initialize_storage(self):
        self.sols_estim = []
        self.sols_true  = []
        return
    
    def recurse_measurements_func(
        self,
        tspan,
        x0_true,
        x0_estim,
        P0,
        t_measurements: list,
        func_simulate_measurements: callable,
        params_measurements: list = None,
        params_measurement_constant: list = [],
        disable_tqdm = True,
    ):
        """Recurse from a function that generates measurements

        The measurement function has signature `y, R = func_simulate_measurements(t,x,params)`. 
        If the measurement could not be generated, return `y = None`.
        
        The `params` argument is used to pass additional parameters: 
        - If each measurement should have a different parameter, provide a list of parameters `params_measurements`.
        - If all measurements should have the same parameter, provide a single parameter `params_measurement_constant`.
        Both `params_measurements` and `params_measurement_constant` cannot be provided. 
        
        Args:
            tspan (list): time span for recursion
            x0_true (np.array): true initial state
            x0_estim (np.array): initial state estimate
            P0 (np.array): initial state covariance matrix
            t_measurements (np.array): measurement times
            func_simulate_ measurements (func): function to simulate measurements, with signature `y, R = func(t, x)`
            params_measurements (list): list of parameters for each measurement
            params_measurement_constant (list): constant parameters for all measurements
            disable_tqdm (bool): whether to disable tqdm progress bar
        """
        assert len(tspan) == 2, "tspan must have length 2"
        assert len(x0_true) == len(x0_estim), "x0_true and x0_estim must have same length"
        if params_measurements is not None:
            assert len(params_measurements) == len(t_measurements), "params_measurements and t_measurements must have same length"
        
        #if (params_measurements is not None) and (params_measurement_constant is not None):
        #    raise ValueError("Only one of params_measurements and params_measurement_constant can be provided")
        
        self._initialize_storage()
        self._initialize_filter(tspan[0], x0_estim, P0)
        self.nx = len(x0_estim)

        # store initial state
        self.ts_y = []
        self.ys = []
        self.Rs = []
        self.Ks = []
        self.ts_update   = [self.filter.t,]
        self.xs_update   = [self.filter.x,]
        self.Ps_update   = [self.filter.P,]

        # iterate over measurement times
        for i,t_meas in tqdm(enumerate(t_measurements), total=len(t_measurements),
            disable = disable_tqdm,
        ):
            # predict until next measurement time
            tspan_i = [self.filter.t, min(t_meas, tspan[1])]

            if tspan_i[1] >= tspan_i[0]:
                # predict state estimate
                sol_estim = self.filter.predict(tspan_i)

                # also propagte true state until next measurement time
                sol_true = self.filter.dynamics.solve(tspan_i, x0_true, stm=False, t_eval = sol_estim.t)
                
                # patch in case the very first measurement is exactly at the initial time of the filter
                if tspan_i[1] == tspan_i[0]:
                    sol_true.t = np.array([tspan_i[0], tspan_i[0]])
                    sol_true.y = np.concatenate((x0_true.reshape(-1,1), x0_true.reshape(-1,1)),axis=1)
                x0_true = sol_true.y[:self.nx,-1]

            # simulate measurement
            if (params_measurements is None) and (params_measurement_constant is None):
                y, R = func_simulate_measurements(t_meas, sol_true.y[:self.nx,-1])
            elif params_measurements is not None:
                y, R = func_simulate_measurements(t_meas, sol_true.y[:self.nx,-1], params_measurements[i])
            elif params_measurement_constant is not None:
                y, R = func_simulate_measurements(t_meas, sol_true.y[:self.nx,-1], params_measurement_constant)
                
            # perform measurement update
            if y is not None:
                _, _, K = self.filter.update(y, R)

            # store information from this iteration
            self.sols_estim.append(sol_estim)
            self.sols_true.append(sol_true)
            self.ts_update.append(self.filter.t)
            self.xs_update.append(copy.deepcopy(self.filter.x))
            self.Ps_update.append(copy.deepcopy(self.filter.P))
            if y is not None:
                self.ts_y.append(self.filter.t)
                self.ys.append(y)
                self.Rs.append(R)
                self.Ks.append(K)
            else:
                self.ts_y.append(np.nan)
                self.ys.append(np.nan)
                self.Rs.append(np.nan)
                self.Ks.append(np.nan)

            # break if final time is exceeded
            if self.filter.t >= tspan[1]:
                break

        # perform final prediction if final measurement is not at final time
        if self.filter.t < tspan[1]:
            tspan_final = [self.filter.t, tspan[1]]
            sol_estim = self.filter.predict(tspan_final)
            self.sols_estim.append(sol_estim)
            self.sols_true.append(self.filter.dynamics.solve(tspan_final, x0_true, stm=False, t_eval = sol_estim.t))
        return
    
    def recurse_measurements_func_events(
        self,
        tspan,
        x0_true,
        x0_estim,
        P0,
        events: list,
        func_simulate_measurements: callable,
        params_measurement_constant: list = None,
        maxiter = 1000,
        meas_interval_min = 0.01,
    ):
        """Recurse from a given set of events when measurements are to be generated
        
        Args:
            tspan (list): time span for recursion
        """
        assert len(tspan) == 2, "tspan must have length 2"
        assert len(x0_true) == len(x0_estim), "x0_true and x0_estim must have same length"
        
        self._initialize_storage()
        self._initialize_filter(tspan[0], x0_estim, P0)
        self.nx = len(x0_estim)

        # store initial state
        self.ts_y = []
        self.ys = []
        self.Rs = []
        self.Ks = []
        self.ts_update   = [self.filter.t,]
        self.xs_update   = [self.filter.x,]
        self.Ps_update   = [self.filter.P,]
        reached_final_time = False

        # iterate over measurement times
        for it in range(maxiter):
            # propagate until event
            _sol_until_event = self.filter.dynamics.solve([self.filter.t, tspan[1]],
                                                          self.filter._x,
                                                          stm = False,
                                                          events = events)
            if _sol_until_event.t[-1] < self.filter.t + meas_interval_min:
                # we further propagate until next event
                _sol_after_event = self.filter.dynamics.solve([self.filter.t, self.filter.t + meas_interval_min],
                                                            self.filter._x,
                                                            stm = False,)
                _sol_until_event = self.filter.dynamics.solve([_sol_after_event.t[-1], tspan[1]],
                                                            _sol_after_event.y[:self.nx,-1],
                                                            stm = False,
                                                            events = events)

            # propagate until next measurement time
            tspan_next_meas = [self.filter.t,
                               min(tspan[1], _sol_until_event.t[-1])]
            sol_estim = self.filter.predict(tspan_next_meas)

            # also propagte true state until next measurement time
            sol_true = self.filter.dynamics.solve([sol_estim.t[0], sol_estim.t[-1]],
                                                  x0_true,
                                                  stm = False,
                                                  t_eval = sol_estim.t)
            x0_true = sol_true.y[:self.nx,-1]

            # store estimate information
            self.sols_estim.append(sol_estim)
            self.sols_true.append(sol_true)

            # exit if final time is reached
            if self.filter.t >= tspan[1]:
                reached_final_time = True
                #print(f"Reached final time {self.filter.t:1.4f} == tspan[1] = {tspan[1]:1.4f} TU")
                break

            # simulate measurement
            if (params_measurement_constant is None):
                y, R = func_simulate_measurements(self.filter.t, sol_true.y[:self.nx,-1])
            else:
                y, R = func_simulate_measurements(self.filter.t, sol_true.y[:self.nx,-1], params_measurement_constant)
                
            # perform measurement update
            if y is not None:
                _, _, K = self.filter.update(y, R)

            # store information from this iteration
            self.ts_update.append(self.filter.t)
            self.xs_update.append(copy.deepcopy(self.filter.x))
            self.Ps_update.append(copy.deepcopy(self.filter.P))
            if y is not None:
                self.ts_y.append(self.filter.t)
                self.ys.append(y)
                self.Rs.append(R)
                self.Ks.append(K)
            else:
                self.ts_y.append(np.nan)
                self.ys.append(np.nan)
                self.Rs.append(np.nan)
                self.Ks.append(np.nan)

        if reached_final_time is False:
            print(f"WARNING : recursion reached max iter before final time!")
        return
    
    def recurse_measurements_list(
        self,
        tspan,
        x0_true,
        x0_estim,
        P0,
        t_measurements: list,
        y_measurements: list,
        R_measurements: list,
        params_measurements: list = None,
        disable_tqdm = True,
    ):
        """Recurse from a given list of measurements
        
        Args:
            tspan (list): time span for recursion
            x0_true (np.array): true initial state
            x0_estim (np.array): initial state estimate
            P0 (np.array): initial state covariance matrix
            t_measurements (list): list of measurement times
            y_measurements (list): list of measurements
            R_measurements (list):list of  measurement covariance matrices
        """
        assert len(tspan) == 2, "tspan must have length 2"
        assert len(x0_true) == len(x0_estim), "x0_true and x0_estim must have same length"
        assert len(t_measurements) == len(y_measurements), "t_measurements and y_measurements must have same length"
        assert len(t_measurements) == len(R_measurements), "t_measurements and R_measurements must have same length"
        if params_measurements is None:
            params_measurements = [None for _ in t_measurements]
        else:
            assert len(t_measurements) == len(params_measurements), "t_measurements and params_measurements must have same length"
        self._initialize_storage()
        self._initialize_filter(tspan[0], x0_estim, P0)
        self.nx = len(x0_estim)

        # store initial state
        self.ts_y = t_measurements          # store to unify interface
        self.ys = y_measurements            # store to unify interface
        self.Rs = R_measurements            # store to unify interface
        self.ts_update   = [self.filter.t,]
        self.xs_update   = [self.filter.x,]
        self.Ps_update   = [self.filter.P,]
        self.Ks = []

        # iterate over measurement times
        for i,(t_meas,y,R) in tqdm(
            enumerate(zip(t_measurements, y_measurements, R_measurements)),
            total=len(t_measurements),
            disable=disable_tqdm,
        ):
            # predict until next measurement time
            tspan_i = [self.filter.t, min(t_meas, tspan[1])]

            if tspan_i[1] > tspan_i[0]:
                # predict state estimate
                sol_estim = self.filter.predict(tspan_i)

                # also propagte true state until next measurement time
                sol_true = self.filter.dynamics.solve(tspan_i, x0_true, stm=False, t_eval = sol_estim.t)
                x0_true = sol_true.y[:self.nx,-1]

            # perform measurement update
            _, _, K = self.filter.update(y, R, params_measurements[i])

            # store information from this iteration
            self.sols_estim.append(sol_estim)
            self.sols_true.append(sol_true)

            self.ts_update.append(self.filter.t)
            self.xs_update.append(copy.deepcopy(self.filter.x))
            self.Ps_update.append(copy.deepcopy(self.filter.P))
            self.Ks.append(K)

            # break if final time is exceeded
            if self.filter.t >= tspan[1]:
                break

        # perform final prediction if final measurement is not at final time
        if self.filter.t < tspan[1]:
            tspan_final = [self.filter.t, tspan[1]]
            sol_estim = self.filter.predict(tspan_final)
            self.sols_estim.append(sol_estim)
            self.sols_true.append(self.filter.dynamics.solve(tspan_final, x0_true, stm=False, t_eval = sol_estim.t))
        return
    
    def plot_state_history(
        self,
        figsize = (12,6),
        lw_true = 1.2,
        lw_estimate = 0.95,
        color_estimate = "crimson",
        color_true = "black",
        TU = 1.0,
        state_multipliers = None,
        state_labels = None,
        time_unit = "TU",
    ):
        """Plot error history of state estimate"""
        nx_half = int(np.ceil(self.nx/2))
        if state_multipliers is None:
            state_multipliers = np.zeros(self.nx)
        else:
            assert len(state_multipliers) == self.nx, "state_multipliers must have length equal to state vector"
        if state_labels is None:
            state_labels = [f"State {i}" for i in range(self.nx)]
        else:
            assert len(state_labels) == self.nx, "state_labels must have length equal to state vector"

        # initialize figure 
        fig, axs = plt.subplots(2,nx_half,figsize=figsize)
        for iax,ax in enumerate(axs.flatten()):
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(f"Time, {time_unit}")
            ax.set_ylabel(state_labels[iax])

        # plot estimate and true history
        for (sol_true, sol_estim) in zip(self.sols_true, self.sols_estim):
            ts = sol_true.t
            x_true = sol_true.y[:self.nx,:]
            x_estim = sol_estim.y[:self.nx,:]

            # plot estimates
            for i in range(nx_half):
                axs[0,i].plot(ts * TU,
                              state_multipliers[i] * x_true[i,:],
                              color = color_true,
                              lw = lw_true)
                axs[1,i].plot(ts * TU, 
                              state_multipliers[i+3] * x_true[i+nx_half,:],
                              color = color_true,
                              lw = lw_true)
                
                axs[0,i].plot(ts * TU,
                              state_multipliers[i] * x_estim[i,:],
                              color = color_estimate,
                              lw = lw_estimate)
                axs[1,i].plot(ts * TU, 
                              state_multipliers[i+3] * x_estim[i+nx_half,:],
                              color= color_estimate,
                              lw = lw_estimate)
        plt.tight_layout()
        return fig, axs
    
    
    def plot_error_history(
        self,
        figsize = (12,6),
        lw_estimate = 0.95,
        color_estimate = "crimson",
        color_sigma = "navy",
        alpha_sigma = 0.3,
        TU = 1.0,
        state_multipliers = None,
        state_labels = None,
        time_unit = "TU",
        k_sigma = 3,
    ):
        """Plot error history of state estimate"""
        nx_half = self.nx//2
        if state_multipliers is None:
            state_multipliers = np.zeros(self.nx)
        else:
            assert len(state_multipliers) == self.nx, "state_multipliers must have length equal to state vector"
        if state_labels is None:
            state_labels = [f"State {i}" for i in range(self.nx)]
        else:
            assert len(state_labels) == self.nx, "state_labels must have length equal to state vector"

        # initialize figure 
        fig, axs = plt.subplots(2,nx_half,figsize=figsize)
        for iax,ax in enumerate(axs.flatten()):
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(f"Time, {time_unit}")
            ax.set_ylabel(state_labels[iax])

        # plot estimate error history
        for (sol_true, sol_estim, P_estim) in zip(self.sols_true, self.sols_estim, self.Ps_update):
            ts = sol_true.t
            x_true = sol_true.y[:self.nx,:]
            x_estim = sol_estim.y[:self.nx,:]

            # a posteriori computation of covariance history
            std_diag = np.zeros((len(ts), self.nx))
            for (idx, (t,y)) in enumerate(zip(sol_estim.t, sol_estim.y.T)):
                if idx == 0:
                    dt = [0.0, 0.0]
                else:
                    dt = [sol_estim.t[idx-1], t]
                Phi_iter = y[self.nx:].reshape(self.nx,self.nx)
                Q_iter = self.filter.func_process_noise(dt, y, self.filter.params_Q)
                P_iter = Phi_iter @ P_estim @ Phi_iter.T + Q_iter
                std_diag[idx,:] = np.sqrt(np.diag(P_iter))

            # plot estimates
            for i in range(nx_half):
                # plot k_sigma bands
                axs[0,i].fill_between(ts * TU,
                                     state_multipliers[i] * (-k_sigma * std_diag[:,i]),
                                     state_multipliers[i] * ( k_sigma * std_diag[:,i]),
                                     color=color_sigma,
                                     alpha=alpha_sigma)
                axs[1,i].fill_between(ts * TU,
                                      state_multipliers[i+3] * (-k_sigma * std_diag[:,i+nx_half]),
                                      state_multipliers[i+3] * ( k_sigma * std_diag[:,i+nx_half]),
                                      color=color_sigma,
                                      alpha=alpha_sigma)
                
                # plot errors
                axs[0,i].plot(ts * TU,
                              state_multipliers[i] * (x_estim[i,:] - x_true[i,:]),
                              color=color_estimate,
                              lw = lw_estimate)
                axs[1,i].plot(ts * TU, 
                              state_multipliers[i+3] * (x_estim[i+nx_half,:] - x_true[i+3,:]),
                              color=color_estimate,
                              lw = lw_estimate)
        plt.tight_layout()
        return fig, axs
    
    def plot_gain_history(
        self,
        figsize = (12,8),
        TU = 1.0,
        time_unit = "TU",
        color = "navy",
    ):
        """Plot gain history
        
        Args:
            figsize (tuple): figure size
            TU (float): time unit
            time_unit (str): time unit string
            color (str): color for plotting

        Returns:
            (tuple): figure and axes
        """
        ms = [len(self.ys[i]) for i in range(len(self.ys))]
        m = next((m_val for m_val in ms if not np.isnan(m_val)), None)
        fig, axs = plt.subplots(self.nx, m, figsize=figsize)
        for ix in range(self.nx):
            for iy in range(m):
                # handle case where we have 1D measurements
                if m == 1:
                    ax = axs[ix]
                else:
                    ax = axs[ix,iy]

                # plot gain history
                ax.grid(True, alpha=0.3)
                ax.plot(self.ts_update[1:], [K[ix,iy] for K in self.Ks],
                                marker="x", color=color)
                ax.set(xlabel=f"Time, {time_unit}", ylabel=f"K[{ix},{iy}]")
        plt.tight_layout()
        return fig, axs