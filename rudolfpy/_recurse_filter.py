"""Object for recursing filter"""

import copy
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ._base_filter import BaseFilter

def plot_state_history(
        history,
        k_sigma = 3,
        figsize = (12,6),
        lw_estimate = 0.95,
        color_estimate = "crimson",
        color_true = "black",
        color_sigma = "navy",
        alpha_sigma = 0.3,
        TU = 1.0,
        state_multipliers = None,
        state_labels = None,
        time_unit = "TU",
    ):
        """Plot state history"""

        xs = np.vstack([item[0] for item in history])
        Ps = np.stack([item[1] for item in history], axis=2)
        ts = np.array([item[2] for item in history])
        dim_x = xs[0].size

        nx_half = dim_x//2
        if state_multipliers is None:
            state_multipliers = np.ones(dim_x)
        else:
            assert len(state_multipliers) == dim_x, "state_multipliers must have length equal to state vector"
        if state_labels is None:
            state_labels = [f"State {i}" for i in range(dim_x)]
        else:
            assert len(state_labels) == dim_x, "state_labels must have length equal to state vector"

        # initialize figure 
        fig, axs = plt.subplots(2,nx_half,figsize=figsize)
        for iax,ax in enumerate(axs.flatten()):
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(f"Time, {time_unit}")
            ax.set_ylabel(state_labels[iax])

            state = xs[:, iax]
            state_sigma = np.sqrt(Ps[iax, iax, :])

            y1 = state - k_sigma * state_sigma
            y2 = state + k_sigma * state_sigma
            

            ax.fill_between(ts * TU,
                            state_multipliers[iax] * y1,
                            state_multipliers[iax] * y2,
                            color=color_sigma,
                            alpha=alpha_sigma)

            ax.plot(ts* TU,
                    state_multipliers[iax] * state,
                    color = color_estimate,
                    lw = lw_estimate)


        plt.tight_layout()
        return fig, axs


def _get_state_history_groundtruth(filter: BaseFilter,
                                   x0: np.ndarray[float],
                                   t_eval: np.ndarray[float]) -> list[tuple]:
        """
        Get state history, using the ground truth initial condition

        Args:
            filter (BaseFilter): a filter object
            x0 (np.ndarray[float]): ground truth initial condition
            t_eval (np.ndarray[float]) : 1d array of times where we would like a state estimate

        Returns:
            target_history (list(tuple)): list of (x, None) state, covariance pairs. We return None because there is no uncertainty propagation
        """

        dim_x = x0.size

        if type(t_eval) is not np.ndarray:
            raise TypeError("`t_eval` must be of type np.ndarray")


        sol_true = filter.dynamics.solve([0.0, t_eval[-1]], x0, t_eval=t_eval)

        target_history = [(sol_true.y[:dim_x, idx], None, t_eval[idx]) for idx in range(t_eval.size)]

        return target_history


def _get_state_history_filter(filter,
                               x0: np.ndarray[float],
                               P0: np.ndarray[float],
                               x0_true: np.ndarray[float],
                               t_eval: np.ndarray[float],
                               t_measurements: np.ndarray[float],
                               params_measurements: list = None,
                               ) -> list[tuple]:
        """
        Get the state history at the requested times with the requested measurements

        Args:
            filter (BaseFilter): filter object
            x0 (np.ndarray): initial state estimate
            P0 (np.ndarray): initial state covariance
            x0_true (np.ndarray): ground truth initial state
            t_eval (np.ndarray[float]): vector of times to query the filtered state at. If None, equally distributed amongst time horizon specified in control
            t_measurements (np.ndarray[float]): vector of times when measurements of state are taken.
            params_measurements (list): list of measurement parameters to pass to `filter.measurement_model.func_simulate_measurements` for each measurement taken.

        Returns:
            target_history (list[tuple[np.ndarray, np.ndarray]]): list of (x, P) state, covariance pairs.

        Notes:
            - if t_measurements is given, params_measurements must have same length as t_measurements. Otherwise, params_measurements is ignored.
        """

        if params_measurements is None:
            params_measurements = [[None]]*t_measurements.size


        if len(params_measurements) != t_measurements.size:
            raise AssertionError(f"params_measurements (size: {len(params_measurements)}) must be of same size as t_measurements (size: f{t_measurements.size}).")
        
        target_history_gt = _get_state_history_groundtruth(filter,x0_true, t_eval=t_measurements)
        target_history = []


        # initialize filter
        filter.initialize(t=0, x0 = x0, P0 = P0)

        # do estimation
        count_m = 0
        count_e = 0
        M = t_measurements.size
        E = t_eval.size
        while count_m < M or count_e < E:
            # compare first two values of two lists, pop the minimum and return whether it is measurement or evaluation
            next_time, type = _get_history_filter_helper(t_measurements, t_eval, count_m, count_e)
            if next_time > filter.t:
                _ = filter.predict(tspan = [filter.t, next_time])

            match type:
                case "eval":
                    x = filter.x
                    P = filter.P
                    t = filter.t
                    target_history.append((deepcopy(x), deepcopy(P), deepcopy(t)))
                    count_e += 1
                case "measure":
                    x_true = target_history_gt[count_m][0]
                    y, R = filter.measurement_model.func_simulate_measurements(filter.t, x_true , params_measurements[count_m])
                    filter.update(y, R, params = params_measurements[count_m])
                    count_m += 1
                case "both":
                    x_true = target_history_gt[count_m][0]
                    y, R = filter.measurement_model.func_simulate_measurements(filter.t, x_true , params_measurements[count_m])
                    filter.update(y, R, params = params_measurements[count_m])
                    x = filter.x
                    P = filter.P
                    t = filter.t
                    target_history.append((deepcopy(x), deepcopy(P), deepcopy(t)))
                    count_m += 1
                    count_e += 1


        return target_history

def _get_history_filter_helper(t_measurements: np.ndarray[float],
                                t_eval: np.ndarray[float],
                                count_m: int,
                                count_e: int) -> tuple[float, str]:
    """
    Helper to evaluate the next time for prediction in the filter and whether to provide update.

    Args:
        t_measurements (np.ndarray[float]): vector of measurement times
        t_eval (np.ndarray[float]): vector of evaluation times
        count_m (int): index of value in t_measurements 
        count_e (int): index of value in t_eval

    Returns
        next_time, type (tuple): tuple containing next time and type (whether to provide update). 
    """

    if (count_m < t_measurements.size) and (count_e < t_eval.size) :
        t_meas = t_measurements[count_m]
        t_ev = t_eval[count_e]

        if t_meas == t_ev:
            type_ = "both"
            next_time = t_meas

        elif t_meas < t_ev:
            type_ = "measure"
            next_time = t_meas

        elif t_meas > t_ev:
            type_ = "eval"
            next_time = t_ev
        else:
            raise ValueError("comparison between floats was not equal, <, or >. Something is wrong...")
    elif (count_m < t_measurements.size) and (count_e >= t_eval.size):
        t_meas = t_measurements[count_m]
        type_ = "measure"
        next_time = t_meas
    elif (count_m >= t_measurements.size) and (count_e < t_eval.size):
        t_ev = t_eval[count_e]
        type_ = "eval"
        next_time = t_ev
    else:
        raise RuntimeError("Control Flow should not reach here, bug in package...")

    return next_time, type_

