"""
    SYNOPSIS
    Implementation of ADMM based layered control architecture for unicycle model with input constraints,
    state constraints, and custom reference states x, y. The code has a class structure

    DESCRIPTION
    Uses cvxpy to solve the r-subproblem and iLQR from trajax to solve the (x, u)-subproblem. Dual
    variables are updated according to the rule specified in Boyd's paper. The dynamics is discretized
    using rk4. The code has been tested for various initial states, maximum speed bounds, horizon and
    granularity of discretization.

    AUTHOR
    Anusha Srikanthan <sanusha@seas.upenn.edu>
"""
import math

import cvxpy as cp
import numpy as np
import jax
import jax.numpy as jnp
from trajax import optimizers
from trajax.integrators import rk4
# from trajax.experimental.sqp import shootsqp, util
import matplotlib.pyplot as plt
from functools import partial
import time
import datetime
import multiprocessing
from admm.model_defn import modified_unicycle, bicycle, dubins


jax.config.update("jax_enable_x64", True)


class admm_lca(object):
    """
    This code is an implementation of a layered control architecture (LCA)
    for any given nonlinear dynamical system.

    Input arguments
    dynamics: dynamics in functional form with arguments (x, u, t)
    T: time horizon
    dt: discretization time step
    m: input dimension
    n: state dimension
    x0: initial condition
    u0: initial input
    goal: final state
    rho: admm parameter for LCA problem
    idx: indices of states to be used as reference variables
    constr box constraints on states

    Member functions
    solve_reference: function to solve one instance of reference planning layer
    rollout: function to compute state trajectory given initial conditions and input
    """
    def __init__(self, dynamics, T, dt, m, n, x0, u0, goal, rho, idx=None, constr=None, umin=None, umax=None):
        self.dynamics = dynamics
        self.dt = dt
        self.x0 = x0
        self.u0 = u0
        self.goal = goal
        self.T = T
        self.n = n
        self.m = m
        self.rho = rho
        # self.parallel = parallel
        if idx is None:
            self.idx = range(self.n)
            self.Tr = np.eye(self.n)
        else:
            self.idx = idx
            self.Tr = np.zeros((self.n, len(self.idx)))
            for k, i in enumerate(self.idx):
                self.Tr[i, k] = 1
            # self.Tr[:len(self.idx), :] = np.eye((len(self.idx)))

        self.r = np.zeros((self.T, len(self.idx)))
        self.a = np.zeros((self.T-1, self.m))
        self.x = np.zeros((self.T, self.n))
        self.u = np.zeros((self.T - 1, self.m))
        self.vr = np.zeros((self.T, len(self.idx)))
        self.vu = np.zeros((self.T - 1, self.m))

        if constr is not None:
            self.constr_idx, self.constr = constr
        else:
            self.constr_idx = None
            self.constr = None
        if umin is not None:
            self.u_min = umin
            self.u_max = umax
        else:
            self.u_min = None
            self.u_max = None

    def check_obstacles(self):
        pass

    def solve_reference_1st(self, seed):
        """
        Member function for solving one instance of planning layer sub-problem in the LCA
        """
        r = cp.Variable(self.r.shape)
        a = cp.Variable(self.a.shape)

        # Cost function for 1st order unicycle model with constraint works well
        # stage_err = cp.hstack([(r[t+1] - r[t]) for t in range(self.T - 1)])
        stage_err = cp.hstack([(a[t+1] - a[t]) for t in range(self.T-2)])

        # Stage error from goal works well for 2nd order model
        # stage_err = cp.hstack([(r[t] - self.goal) for t in range(self.T)])
        final_err = r[-1] - self.goal

        stage_cost = 0.1 * cp.sum_squares(stage_err)
        final_cost = 1000 * cp.sum_squares(final_err)
        # final_cost = cp.sum_squares(final_err)
        # utility_cost = cp.sum_squares(err)
        utility_cost = stage_cost + final_cost
        admm_cost = (self.rho / 2) * cp.sum_squares(r - self.x @ self.Tr + self.vr) + (self.rho / 2) * cp.sum_squares(a - self.u + self.vu)

        # Adding constraints seems to work well for 1st order but not 2nd order
        # constr = [r[0] == self.x0[0:len(self.idx)], r[-1] == self.goal]
        # constr = [r[0] == self.x0[0:len(self.idx)]]
        constr = []
        # constr = [r[0] == self.x0[0:len(self.idx)], r[-1] == self.goal]
        if seed is not None:
            for i in range(len(seed)):
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #do not change the resolution here
                constr.append(cp.norm(r[int((i + 1) * self.T / (len(seed) + 1)) + 1][0:2] - seed[i]) <= 0.2)
                # constr.append(cp.norm(r[int(self.T/2)+1] - seed) <= 0.1)
        constr.append(r[-1] == self.goal)
        for i in range(self.T - 1):
            constr.append(a[i] <= self.u_max)
            constr.append(a[i] >= self.u_min)

        # Corridor constraints
        # if self.constr:
        #     # for k in self.constr_idx:
        #     # Need to make state constraints more general
        #     constr.append(r[int(self.T/2):, 1] >= self.constr[1][0])
        #     constr.append(r[int(self.T / 2):, 1] <= self.constr[1][1])
        #     constr.append(r[:-int(self.T/2), 0] >= self.constr[0][0])
        #     constr.append(r[:-int(self.T / 2), 0] <= self.constr[0][1])

        # State constraints
        if self.constr:
            for k, idx in enumerate(self.constr):
                constr.append(r[:, self.constr_idx[k]] <= idx[1])
                constr.append(r[:, self.constr_idx[k]] >= idx[0])
        ref_prob = cp.Problem(cp.Minimize(utility_cost + admm_cost), constr)
        ref_prob.solve(solver=cp.MOSEK)
        # try:
        #     ref_prob.solve()
        # except cp.error.SolverError:
        #     ref_prob.solve(solver=cp.SCS)
        self.r = r.value
        self.a = a.value

    def solve_reference_2nd(self, seed):
        """
        Member function for solving one instance of planning layer sub-problem in the LCA
        """
        r = cp.Variable(self.r.shape)
        a = cp.Variable(self.a.shape)

        # Cost function for 1st order unicycle model with constraint works well
        # stage_err = cp.hstack([(r[t+1] - r[t]) for t in range(self.T - 1)])

        # Stage error from goal works well for 2nd order model
        stage_err = cp.hstack([(r[t] - self.goal) for t in range(self.T)])
        final_err = r[-1] - self.goal

        stage_cost = 0.1 * cp.sum_squares(stage_err)
        final_cost = 1000 * cp.sum_squares(final_err)
        # final_cost = cp.sum_squares(final_err)
        # utility_cost = cp.sum_squares(err)
        utility_cost = stage_cost + final_cost
        admm_cost = (self.rho / 2) * cp.sum_squares(r - self.x @ self.Tr + self.vr) + (self.rho / 2) * cp.sum_squares(a - self.u + self.vu)
        # Adding constraints seems to work well for 1st order but not 2nd order
        # constr = [r[0] == self.x0[0:len(self.idx)], r[-1] == self.goal]
        # constr = [r[0] == self.x0[0:len(self.idx)]]
        constr = []
        if seed is not None:
            for i in range(len(seed)):
                constr.append(cp.norm(r[int((i+1)*self.T/(len(seed)+1))+1][0:2] - seed[i]) <= 0.3)
        constr.append(r[-1] == self.goal)
        for i in range(self.T - 1):
            constr.append(a[i] <= self.u_max)
            constr.append(a[i] >= self.u_min)

        # Corridor constraints
        # if self.constr:
        #     # for k in self.constr_idx:
        #     # Need to make state constraints more general
        #     constr.append(r[int(self.T/2):, 1] >= self.constr[1][0])
        #     constr.append(r[int(self.T / 2):, 1] <= self.constr[1][1])
        #     constr.append(r[:-int(self.T/2), 0] >= self.constr[0][0])
        #     constr.append(r[:-int(self.T / 2), 0] <= self.constr[0][1])

        # State constraints
        if self.constr:
            for k, idx in enumerate(self.constr):
                constr.append(r[:, self.constr_idx[k]] <= idx[1])
                constr.append(r[:, self.constr_idx[k]] >= idx[0])
        ref_prob = cp.Problem(cp.Minimize(utility_cost + admm_cost), constr)
        # try:
        #     ref_prob.solve(solver=cp.CLARABEL)
        # except cp.error.SolverError:
        #     ref_prob.solve(solver=cp.ECOS)
        ref_prob.solve(solver=cp.ECOS)

        self.r = r.value
        self.a = a.value

    def rollout(self):
        """
        Member function to compute state trajectory
        """
        # self.x[0] = self.x0
        return optimizers.rollout(self.dynamics, self.u, self.x0)
        # self.x = optimizers.rollout(dynamics, U, x0)
        # for t in range(self.T - 1):
        #     self.x[t + 1] = self.dynamics(self.x[t], self.u[t], t)
        # return self.x

    def plot_car(self, car_len, i=None, num_plots=None, col='black', col_alpha=1):
        w = car_len / 2
        x = np.zeros(self.n)
        if num_plots == -1:
            x[0:2] = self.goal[:2]
            x[2] = 0
        else:
            x[0:2] = self.x[int(i * (self.T+1)/num_plots), :2]
            if self.n == 5:
                # x[2] = self.x[int(i * (self.T+1)/num_plots), 4]
                x[2] = self.x[int(i * (self.T + 1) / num_plots), 3]
            else:
                x[2] = self.x[int(i * (self.T+1)/num_plots), 2]
        x_rl = x[:2] + 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
        x_rr = x[:2] - 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
        x_fl = x_rl + car_len * np.array([np.cos(x[2]), np.sin(x[2])])
        x_fr = x_rr + car_len * np.array([np.cos(x[2]), np.sin(x[2])])
        x_plot = np.concatenate((x_rl, x_rr, x_fr, x_fl, x_rl))
        plt.plot(x_plot[0::2], x_plot[1::2], linewidth=2, c=col, alpha=col_alpha)
        plt.scatter(x[0], x[1], marker='.', s=200, c=col, alpha=col_alpha)


@partial(jax.jit, static_argnums=0)
def ctl_prob(dynamics, x0, r, a, vr, vu, u0, rho, T, Tr):
    def cost(x, u, t):
        # state_err = state_wrap(r[t] - Tr @ x + vr[t])
        state_err = r[t] - x @ Tr + vr[t]
        input_err = a[t] - u + vu[t]
        stage_costs = ((rho / 2) * jnp.dot(state_err, state_err) +
                           (rho / 2) * jnp.dot(input_err, input_err) + 0.01 * jnp.dot(u, u))
        final_costs = rho / 2 * jnp.dot(state_err, state_err)
        return jnp.where(t == T, final_costs, stage_costs)

    X, U, _, _, alpha, lqr_val, _ = optimizers.ilqr(
            cost,
            dynamics,
            x0,
            u0,
            maxiter=10
    )

    return X, U, lqr_val

    # To use constrained ilqr, uncomment this part of the code and comment above
    # def eq_constr(x, u, t):
    #     del u
    #     def goal_constr(x):
    #         err = x[0:2] - r[-1]
    #         return err
    #     return jnp.where(t == T, goal_constr(x), np.zeros(u0.shape[1]))
    #
    # sol = optimizers.constrained_ilqr(cost, dynamics, x0, u0, equality_constraint=eq_constr, maxiter_ilqr=10, maxiter_al=10)

    # return sol[0], sol[1], None


def run_parallel_admm(admm_obj, num_samples=5, dispersion_size=0.3, dyn="2nd", solver=ctl_prob, tol=0.1,num_wayp=3,use_points= False,points=None, horizon=200):
    parallel = False
    dis_vector = admm_obj.goal[0:2]-admm_obj.x0[0:2]
    if points is not None:
        samples = points
        num_samples = len(points)
    else:
        normal = np.array([dis_vector[1],-dis_vector[0]])
        normal = normal/np.linalg.norm(normal)
        # samples = np.linspace(midpt + np.array([0, dispersion_size]),
        #                       midpt - np.array([0, dispersion_size]), num_samples)
        samples = np.zeros((num_samples,num_wayp,2))
        mid_id = int((num_samples-1)/2)
        for j in range(num_wayp):
            midpt = admm_obj.x0[0:2]+(j+1)*(-admm_obj.x0[0:2] + admm_obj.goal[0:2]) / (num_wayp+1)
            samples[mid_id,j] = midpt
            for i in range(mid_id):
                samples[mid_id-(i+1),j] = midpt+dispersion_size*(i+1)*normal
                samples[mid_id+(i+1),j] = midpt-dispersion_size*(i+1)*normal
    print("Waypoints", samples)
    nominal_ctrl = []
    # extra_time = []
    if not parallel:
        # for _ in samples:
        for i in range(num_samples):
            if i==0 or i==num_samples-1:
                nominal_ctrl.append(run_admm(admm_obj, samples[i], dyn, solver, tol,int(horizon)))
            else:
                nominal_ctrl.append(run_admm(admm_obj, samples[i], dyn, solver, tol,int(horizon)))

            # nominal_ctrl.append(run_admm(admm_obj, samples[i], dyn, solver, tol,int(admm_obj.T+abs(i-mid_id)*25)))
    # Use multiprocessing to run multiple simulations in parallel.

    else:

        num_cores = min(multiprocessing.cpu_count(), 40)

        print(
            "Running {} simulations in parallel with up to {} cores.".format(
                num_samples, num_cores
            )
        )

        pool = multiprocessing.Pool(num_cores)

        manager = multiprocessing.Manager()

        lock = manager.Lock()

        def update_results(result):
            with lock:
                nominal_ctrl.append(result)

        code_rate = 0.1  # simulations per second, empirically determined.
        expected_duration_seconds = num_samples / code_rate

        current_time = datetime.datetime.now()
        end_time = current_time + datetime.timedelta(seconds=expected_duration_seconds)

        print(f"Start time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            "Expected duration: %3.2f seconds (%3.2f minutes, or %3.2f hours)"
            % (
                expected_duration_seconds,
                expected_duration_seconds / 60,
                expected_duration_seconds / 3600,
            )
        )
        print(
            f"Program *may* end around, depending on number of samples, etc.: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        print("Running simulations (in parallel)...")
        for _ in samples:
            pool.apply_async(
                run_admm,
                args=(
                    admm_obj,
                    _,
                    dyn,
                    solver,
                    tol,
                ),
                callback=update_results,
            )

        pool.close()
        pool.join()

    return nominal_ctrl


def run_admm(admm_obj, seed=None, dyn="1st", solver=ctl_prob, tol=0.1e-1, duration=None):
    """
    Function to run admm iterations until desired tolerance is achieved
    :param admm_obj: class object that contains details of control problem
    :param u_max: maximum input allowed
    :param tol: error tolerance for admm iterations
    :return:
    :param gain_K: gain for converged lqr iteration
    :param gain_k: solution from final lqr iteration
    :param admm_obj.x: final state trajectory
    :param admm_obj.u: final input trajectory
    :param admm_obj.r: final reference trajectory
    """
    if duration:
        admm_obj.T = duration
        T = duration
    else:
        T = admm_obj.T
    # print(T)
    # exit()

    admm_obj.r = np.zeros((admm_obj.T, len(admm_obj.idx)))
    admm_obj.a = np.zeros((admm_obj.T-1, admm_obj.m))
    admm_obj.x = np.zeros((admm_obj.T, admm_obj.n))
    admm_obj.u = np.zeros((admm_obj.T - 1, admm_obj.m))
    admm_obj.vr = np.zeros((admm_obj.T, len(admm_obj.idx)))
    admm_obj.vu = np.zeros((admm_obj.T - 1, admm_obj.m))
    n = admm_obj.n
    dynamics = admm_obj.dynamics
    x0 = admm_obj.x0
    r = np.array(admm_obj.r)
    a = np.array(admm_obj.a)
    vr = np.array(admm_obj.vr)
    vu = np.array(admm_obj.vu)
    u = np.array(admm_obj.u)
    rho = admm_obj.rho
    if n > r.shape[1]:
        Tr = admm_obj.Tr
    else:
        Tr = np.eye(n)
    # print(u)
    X, U, _ = solver(dynamics, x0, r, a, vr, vu, u, rho, T, Tr)
    admm_obj.x = np.array(X)
    admm_obj.u = np.array(U)

    k = 0
    err = 100
    start = time.time()
    # while err >= tol:
    pr_res_norm = 100
    dual_res_norm = 100
    eps_pri = 0
    eps_dual = 0
    while (pr_res_norm >= eps_pri) and (dual_res_norm >= eps_dual):
        if k > 400:
            tol += 0.1
        # update r
        # if k % 3 == 0:
        #     admm_obj.solve_reference()

        if dyn == "1st":
            admm_obj.solve_reference_1st(seed)
        else:
            admm_obj.solve_reference_2nd(seed)

        k += 1

        # update x u
        prev_x = admm_obj.x
        prev_u = admm_obj.u

        r = np.array(admm_obj.r)
        a = np.array(admm_obj.a)
        vr = np.array(admm_obj.vr)
        vu = np.array(admm_obj.vu)
        u = np.array(admm_obj.u)
        rho = admm_obj.rho

        if solver is not None:
            X, U, lqr_val = solver(dynamics, x0, r, a, vr, vu, u, rho, T, Tr)
        else:
            X, U, lqr_val = ctl_prob(dynamics, x0, r, a, vr, vu, u, rho, T)
        admm_obj.x = np.array(X)
        admm_obj.u = np.array(U)

        # compute residuals
        sxk = admm_obj.rho * (prev_x - admm_obj.x).flatten()
        suk = admm_obj.rho * (prev_u - admm_obj.u).flatten()
        dual_res_norm = np.linalg.norm(np.hstack([sxk, suk]))
        rxk = (admm_obj.r - admm_obj.x @ admm_obj.Tr).flatten()
        auk = (admm_obj.a - admm_obj.u).flatten()
        pr_res_norm = np.linalg.norm(np.hstack([rxk, auk]))
        # pr_res_norm = np.linalg.norm(admm_obj.r - admm_obj.x @ admm_obj.Tr) #+ np.linalg.norm(admm_obj.a - admm_obj.u)

        # update rhok and rescale vk
        if pr_res_norm > 10 * dual_res_norm:
            admm_obj.rho = 2 * admm_obj.rho
            admm_obj.vr = admm_obj.vr / 2
            admm_obj.vu = admm_obj.vu / 2
        elif dual_res_norm > 10 * pr_res_norm:
            admm_obj.rho = admm_obj.rho / 2
            admm_obj.vr = admm_obj.vr * 2
            admm_obj.vu = admm_obj.vu * 2

        # admm_obj.u = np.where(admm_obj.u >= u_max, u_max, admm_obj.u)
        # admm_obj.u = np.where(admm_obj.u <= u_min, u_min, admm_obj.u)
        admm_obj.vr = admm_obj.vr + admm_obj.r - admm_obj.x @ admm_obj.Tr
        admm_obj.vu = admm_obj.vu + admm_obj.a - admm_obj.u

        err = np.trace((admm_obj.r - admm_obj.x @ admm_obj.Tr).T @ (admm_obj.r - admm_obj.x @ admm_obj.Tr)) + np.sum(
            (admm_obj.a - admm_obj.u).T @ (admm_obj.a - admm_obj.u))

        eps_pri = np.sqrt(admm_obj.n + admm_obj.m) * tol + 0.001 * max(np.linalg.norm(np.hstack([(admm_obj.x @ admm_obj.Tr).flatten(), (admm_obj.u).flatten()])), np.linalg.norm(np.hstack([(admm_obj.r).flatten(), (admm_obj.a).flatten()])))
        eps_dual = np.sqrt(admm_obj.n + admm_obj.m) * tol + 0.001 * np.linalg.norm(np.hstack([(admm_obj.vr).flatten(), (admm_obj.vu).flatten()]))

        # err = np.linalg.norm(admm_obj.x[-1] @ admm_obj.Tr - admm_obj.goal)

        # print("Err", )

    end = time.time()

    print("Time", end - start)

    # if np.linalg.norm(admm_obj.r[0] - admm_obj.x[0][0:len(admm_obj.idx)]) < 0.01:
    #     print("Same init cond")
    # else:
    #     print(admm_obj.r[0])
    #     print("init cond", admm_obj.x[0][0:len(admm_obj.idx)])

    Q, Qq, R, Rr, M, A, B = lqr_val
    gain_K, gain_k, _, _ = optimizers.tvlqr(Q, Qq, R, Rr, M, A, B, np.zeros((T, n)))
    return gain_K, gain_k, admm_obj.x, admm_obj.u, admm_obj.r, A, B
    # return None, None, admm_obj.x, admm_obj.u, admm_obj.r


def test_car(dynamics_model, dyn, T, dt, m, n, goal, x0, u0, u_max, u_min, rho, idx, constr=None, filename="car.png", parallel=False):
    """
    Function to test LCA on a dynamics simulation of a car
    :param dynamics_model: Specified dynamics model
    :param T: time horizon
    :param dt: discretization time
    :param m: input dimension
    :param n: state dimension
    :param goal: goal
    :param x0: initial state
    :param u0: initial input trajectory
    :param u_max: maximum allowable inputs
    :param rho: initial admm parameter
    :param constr: state constraints
    :return: None
    """
    # Discretize unicycle using rk4 - simple dynamics
    dynamics = rk4(dynamics_model, dt=dt)
    admm_obj = admm_lca(dynamics, T, dt, m, n, x0, u0, goal, rho, idx, constr, u_min, u_max)

    if parallel is True:
        start = time.time()
        nominal_ctrl = run_parallel_admm(admm_obj, num_samples=5, dyn=dyn)
        end = time.time()
        print("Time taken", end - start)
        print(len(nominal_ctrl))

        # Plotting figures
        # ==================================================================================================================
        plt.figure()
        # plt.axis("off")
        plt.gca().set_aspect('equal', adjustable='box')
        for i in range(5):
            gain_K, gain_k, x, u, r, _, _ = nominal_ctrl[i]
            plt.plot(x[:, 0], x[:, 1])

            plt.gca().set_aspect('equal', adjustable='box')

            car_len = 0.25
            nb_plots = T
            for i in range(nb_plots):
                admm_obj.plot_car(car_len, i, nb_plots, 'black', 0.1 + 0.9 * i / nb_plots)
            admm_obj.plot_car(car_len, -1, T + 1, 'black')
            admm_obj.plot_car(car_len, -1, -1, 'red')

            # x = admm_obj.rollout()
            # plt.plot(x[:, 0], x[:, 1], c='black')


            # plt.plot(r[:, 0], r[:, 1])



            # car_len = 0.25
            # nb_plots = T
            # for i in range(nb_plots):
            #     admm_obj.plot_car(car_len, i, nb_plots, 'black', 0.1 + 0.9 * i / nb_plots)
            # admm_obj.plot_car(car_len, -1, T + 1, 'black')
            # admm_obj.plot_car(car_len, -1, -1, 'red')

            # x = admm_obj.rollout()
            # plt.plot(x[:, 0], x[:, 1], c='black')
            # print("Before rollout", np.linalg.norm(x))
            # x = admm_obj.rollout()
            # print("After rollout", np.linalg.norm(x))

        plt.scatter(admm_obj.goal[0], admm_obj.goal[1], color='r', marker='.', s=200, label="Desired pose")
            # Corridor constraints
            # if constr:
            #     plt.plot(np.array([1, 1]), np.array([0, 1.5]), color='b', linewidth=4)
            #     plt.plot(np.array([1, 4]), np.array([1.5, 1.5]), color='b', linewidth=4)
            #     plt.plot(np.array([0, 0]), np.array([0, 2.5]), color='b', linewidth=4)
            #     plt.plot(np.array([0, 4]), np.array([2.5, 2.5]), color='b', linewidth=4)
        plt.legend()
            # plt.savefig("../examples/" + filename)
        plt.show()

    else:
        gain_K, gain_k, x, u, r, _, _ = run_admm(admm_obj, seed=None, dyn=dyn, solver=ctl_prob)

        while np.linalg.norm(r[-1] - goal) >= 0.2:
            T += 10
            admm_obj = admm_lca(dynamics, T, dt, m, n, x0, u0, goal, rho, idx, constr, u_min, u_max)
            gain_K, gain_k, x, u, r, _, _ = run_admm(admm_obj, seed=None, dyn=dyn, solver=ctl_prob)
        print("Final T", T)
        # print("Inputs", u)
        plt.plot(x[:, 0], x[:, 1])
        plt.plot(r[:, 0], r[:, 1])

        # Plotting figures
        # ==================================================================================================================
        plt.figure()
        # plt.axis("off")
        plt.gca().set_aspect('equal', adjustable='box')

        car_len = 0.25
        nb_plots = T
        for i in range(nb_plots):
            admm_obj.plot_car(car_len, i, nb_plots, 'black', 0.1 + 0.9 * i / nb_plots)
        admm_obj.plot_car(car_len, -1, T + 1, 'black')
        admm_obj.plot_car(car_len, -1, -1, 'red')

        x = admm_obj.rollout()
        plt.plot(x[:, 0], x[:, 1], c='black')
        # print("Before rollout", np.linalg.norm(x))
        # x = admm_obj.rollout()
        # print("After rollout", np.linalg.norm(x))

        plt.scatter(admm_obj.goal[0], admm_obj.goal[1], color='r', marker='.', s=200, label="Desired pose")
        # Corridor constraints
        # if constr:
        #     plt.plot(np.array([1, 1]), np.array([0, 1.5]), color='b', linewidth=4)
        #     plt.plot(np.array([1, 4]), np.array([1.5, 1.5]), color='b', linewidth=4)
        #     plt.plot(np.array([0, 0]), np.array([0, 2.5]), color='b', linewidth=4)
        #     plt.plot(np.array([0, 4]), np.array([2.5, 2.5]), color='b', linewidth=4)
        plt.legend()
        plt.savefig("../examples/"+filename)
        plt.show()


def main():
    # Define variables for the problem specification
    T = 50
    dt = 0.1
    goal = np.array([4, 1.5])
    u_max = np.array([1, 4])
    u_min = np.array([0, -4])

    # choose 0.1 to prevent backward velocity
    # u_min = np.array([0.1, -4])

    # Sample the initial condition from a random normal distribution
    # np.random.seed(0)
    # rng = np.random.default_rng()
    # x0 = rng.standard_normal(n)
    x0 = np.array([0, 0, 0.2])
    u0 = np.zeros(2)
    rho = 50
    m = 2
    n = 3

    # A = np.diag(np.ones(n)) + np.diag(np.ones(n-1),1)
    # B = np.zeros((n, m))
    # B[-m:, :] = np.eye(m)



    # # Linear system dynamics
    # def linsys(x, u, t):
    #     return A @ x + B @ u
    #
    # test_car(linsys, T, dt, m, n, np.ones(n), x0, u0, u_max, u_min, rho, None, None, filename="linear.png")

    # wrap to [-pi, pi]
    def angle_wrap(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi


    def car(x, u, t):
        px, py, theta = x
        v, omega = u
        theta = angle_wrap(theta)
        return jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), omega])

    # Heuristic for number of samples

    # test_car(car, "1st", T, dt, m, n, goal, x0, u0, u_max, u_min, rho, (0, 1), filename="unicycle.png")
    # import pdb; pdb.set_trace()
    test_car(car, "1st", T, dt, m, n, goal, x0, u0, u_max, u_min, rho, (0, 1), filename="unicycle.png", parallel=True)
    test_car(modified_unicycle, "1st", T, dt, m, n, goal, x0, u0, u_max, u_min, rho, (0, 1), filename="mod_unicycle.png", parallel=True)

    # constr = ((0, 1), [[0, 1], [1.5, 2.5]])
    # for i in range(5):
    #     goal = goal + i * np.ones(goal.shape)
    #     test_car(car, T, dt, m, n, goal, x0, u0, u_max, u_min, rho, (0, 1), filename="unicycle.png")
    #     # test_car(car, T, dt, m, n, goal, x0, u0, u_max, u_min, rho, (0, 1), constr, "unicycle.png")
    #     x0[0:2] = goal

    # def angle_wrap(theta):
    #     return theta % (2 * np.pi)

    # Dubins car dynamics model
    def complex_car(x, u, t):
        """
        States : x, y, v, psi, omega
        Control: ul, ua - linear and angular acceleration
        Let m = 1, Iz = 1, a = 0.5
        :return:
        """
        px, py, v, psi, w = x
        # px, py, v, w, psi = x
        psi = angle_wrap(psi)
        ul, ua = u
        # return jnp.array([v * jnp.cos(psi) - 0.01 * w * jnp.sin(psi), v * jnp.sin(psi) + 0.01 * w * jnp.cos(psi), ul - 0.01 * w ** 2, ua, w])
        return jnp.array([v * jnp.cos(psi) - 0.01 * w * jnp.sin(psi), v * jnp.sin(psi) + 0.01 * w * jnp.cos(psi), ul - 0.01 * w ** 2, w, ua])

    n = 5
    T = 50
    # np.random.seed(0)
    # rng = np.random.default_rng()
    # x0 = rng.standard_normal(n)
    x0 = np.zeros(n)
    x0[4] = 0
    x0[2] = 1
    # x0 = np.array([4, 3, 1, 0.2, 0])
    goal = np.array([3, 1.5, 0, 0])
    u_max = np.array([0.74, 2.2])
    u_min = np.array([-0.74, -2.2])
    constr = [(2, 3), ([-1, 1], [-4, 4])]
    # test_car(complex_car, "2nd", T, dt, m, n, goal, x0, u0, u_max, u_min, rho, (0, 1, 2, 4), constr, "dubins_car.png")
    test_car(complex_car, "2nd", T, dt, m, n, goal, x0, u0, u_max, u_min, rho, (0, 1, 2, 4), constr, "dubins_car.png", parallel=True)
    # test_car(complex_car, T, dt, m, n, goal, x0, u0, u_max, rho, constr, "dubins_car.png")


if __name__ == '__main__':
    main()


