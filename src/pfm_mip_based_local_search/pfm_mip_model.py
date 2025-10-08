import sys

import docplex.mp.model as cpx
import numpy as np

from .pfm_problem_definition import PFMproblem


# Class to build PFM MIP formulation model using docplex
class PFMmip(PFMproblem):

    # Constructor
    def __init__(
        self,
        data_file: str = None,
        inst_name: str = None,
        n: int = None,
        m: int = None,
        proc_times: list = None,
        best: int = None,
    ):
        super().__init__(data_file, inst_name, n, m, proc_times, best)

    # Build MIP or LP relaxation
    def build_model(self, lp_relaxed: bool = False):
        self.pfm = cpx.Model("Stafford_MIP")
        self.add_variables(lp_relaxed)
        self.add_obj_function()
        self.add_job_assignment_const()
        self.add_pos_assignment_const()
        self.add_finishing_first_const()
        self.add_pos_finishing_const()
        self.add_mach_finishing_const()

    # Add variables
    def add_variables(self, lp_relaxed):
        if lp_relaxed:
            self.xs = {
                (j, k): self.pfm.continuous_var(name=f"x_{j}_{k}", lb=0)
                for j in self.jobs
                for k in self.seq
            }
        else:
            self.xs = {
                (j, k): self.pfm.binary_var(name=f"x_{j}_{k}")
                for j in self.jobs
                for k in self.seq
            }

        self.fs = {
            (i, k): self.pfm.continuous_var(name=f"f_{i}_{k}", lb=0)
            for i in self.machines
            for k in self.seq
        }

    # Add objective function: makespan
    def add_obj_function(self):
        self.pfm.minimize(self.fs[self.m, self.n])

    # Add job assignment constraints
    def add_job_assignment_const(self):
        for j in self.jobs:
            expr = self.pfm.linear_expr()
            for k in self.seq:
                expr.add_term(self.xs[j, k], 1)

            self.pfm.add(self.pfm.eq_constraint(expr, 1))

    # Add position assignment constraints
    def add_pos_assignment_const(self):
        for k in self.seq:
            expr = self.pfm.linear_expr()
            for j in self.jobs:
                expr.add_term(self.xs[j, k], 1)

            self.pfm.add(self.pfm.eq_constraint(expr, 1))

    # Add finishing first position constraints
    def add_finishing_first_const(self):
        expr = self.pfm.linear_expr()
        expr.add_term(self.fs[1, 1], 1)
        for j in self.jobs:
            expr.add_term(self.xs[j, 1], -self.proc_times[0, j - 1])

        self.pfm.add(self.pfm.eq_constraint(expr, 0))

    # Add position-based finishing times constraints
    def add_pos_finishing_const(self):
        for i in self.machines:
            for k in self.seq[:-1]:
                expr = self.pfm.linear_expr()
                expr.add_term(self.fs[i, k + 1], 1)
                expr.add_term(self.fs[i, k], -1)

                for j in self.jobs:
                    expr.add_term(self.xs[j, k + 1], -self.proc_times[i - 1, j - 1])

                self.pfm.add(self.pfm.ge_constraint(expr, 0))

    # Add machine-based finishing times constraints
    def add_mach_finishing_const(self):
        for i in self.machines[:-1]:
            for k in self.seq:
                expr = self.pfm.linear_expr()
                expr.add_term(self.fs[i + 1, k], 1)
                expr.add_term(self.fs[i, k], -1)

                for j in self.jobs:
                    expr.add_term(self.xs[j, k], -self.proc_times[i, j - 1])

                self.pfm.add(self.pfm.ge_constraint(expr, 0))

    # Set CPLEX model parameters:
    # MIP emphasis: {0: 'BALANCED', 1: 'FEASIBILITY', 2: 'OPTIMALITY', 3: 'BESTBOUND', 4: 'HIDDENFEAS'},
    # MIP solution limit (any positive integer) and Time limit (any nonnegative value in seconds 'wall clock time')
    def set_model_parameters(
        self,
        emphasis: int = 0,
        mip_solution_limit: int = None,
        time_limit: int = None,
        n_threads: int = None,
    ):
        """Configure CPLEX solver parameters for the MIP model.

        Sets various CPLEX parameters to control solver behavior, performance,
        and computational limits for the Permutation Flow Shop MIP formulation.

        Args:
            emphasis (int, optional): MIP emphasis setting controlling the trade-off
                between feasibility and optimality. Defaults to 0 (balanced).
                - 0: Balanced approach
                - 1: Emphasize feasibility over optimality
                - 2: Emphasize optimality over feasibility
                - 3: Emphasize best bound improvement
                - 4: Emphasize finding hidden feasible solutions
            mip_solution_limit (int, optional): Maximum number of MIP solutions
                to find before terminating. If None, no limit is set.
            time_limit (int, optional): Maximum wall clock time in seconds
                before terminating the solver. If None, no time limit is set.
            n_threads (int, optional): Number of parallel threads for solver.
                If None, CPLEX uses default thread count. Setting to -1 enables
                opportunistic parallel mode.

        Note:
            This method must be called after build_model() and before solve_model().
            Parameters are applied to the CPLEX model instance stored in self.pfm.
            Wall clock time is used for time_limit (clocktype=2).
        """

        # Set the feasibility-optimality emphasis
        if emphasis != 0:
            self.pfm.parameters.emphasis.mip = emphasis

        # Set the number of MIP solutions limit
        if mip_solution_limit is not None:
            self.pfm.parameters.mip.limits.solutions = mip_solution_limit

        # Set the time limit (real-elapsed time)
        if time_limit is not None:
            self.pfm.parameters.clocktype = 2  # Wall clock time
            self.pfm.parameters.timelimit = time_limit

        # Change to parallelisation and set number of threads
        if n_threads is not None:
            self.pfm.parameters.parallel = -1
            self.pfm.parameters.threads = n_threads

    # Solve MIP/LP model.
    def solve_model(self, show_log: bool = False):
        """Solve the MIP formulation using CPLEX solver.

        Executes the CPLEX solver on the previously built PFM MIP model and
        returns the optimal or best-found makespan value.

        Args:
            show_log (bool, optional): If True, displays solver output and
                progress information during optimization. Defaults to False.

        Returns:
            int: The makespan value (objective function value) rounded up to
                the nearest integer. This represents the completion time of
                the last job on the last machine.

        Raises:
            NameError: If the model has not been built before calling this method.
                Call build_model() first.
            ValueError: If the MIP solution is infeasible or returns a null value.
                This may occur when the problem constraints are contradictory
                or when solver parameters are too restrictive.

        Note:
            The method uses clean_before_solve=True to ensure a fresh solve.
            The makespan value is printed to console for monitoring purposes.
            The solution is stored in self.pfm_sol for further analysis.
        """

        try:
            self.pfm
        except NameError:
            print("The model needs to be defined before solving it!")
            sys.exit()

        self.pfm_sol = self.pfm.solve(clean_before_solve=True, log_output=show_log)

        if self.pfm_sol is None:
            raise ValueError(
                f"PFM solution not found for {self.data_file}! Solution status: {self.pfm.solve_status}, {self.pfm.solve_details}"
            )
        else:
            c_max = np.ceil(self.pfm_sol.get_objective_value())
            print("makespan value = ", c_max)

        return c_max

    # Get variable values
    def get_var_values(self):

        try:
            self.pfm_sol
        except NameError:
            print("The model needs to be solved first!")
            sys.exit()

        xs_vars = self.get_x_var_vals()
        fs_vars = self.get_f_var_vals()

        return {"x_vars": xs_vars, "f_vars": fs_vars}

    # Get x variable values
    def get_x_var_vals(self):

        xs_vars = {}
        for x_var in self.xs.values():
            var_val = self.pfm_sol.get_value(x_var)
            xs_vars[x_var.name] = var_val

        return xs_vars

    # Get f variable values
    def get_f_var_vals(self):

        fs_vars = {}
        for f_var in self.fs.values():
            var_val = self.pfm_sol.get_value(f_var)
            fs_vars[f_var.name] = var_val

        return fs_vars

    # Add fixed variable constraints for the 'relax-and-fix' algorithm -> X_{jk} = 1
    def add_fixed_var_const(self, j: int, k: int, const_name: str = None):

        expr = self.pfm.linear_expr()
        expr.add_term(self.xs[j, k], 1)
        self.pfm.add(self.pfm.eq_constraint(expr, 1), const_name)

    # Add fixed variable constraints for the local search algorithm -> X_{jk} = 0
    def add_fixed_var_const_zero(self, j: int, k: int, const_name: str = None):

        expr = self.pfm.linear_expr()
        expr.add_term(self.xs[j, k], 1)
        self.pfm.add(self.pfm.eq_constraint(expr, 0), const_name)

    # Add MIP start (warm start)
    def add_mip_solution(self, xs_vars: dict):

        self.pfm.clear_mip_starts()
        mip_start = self.pfm.new_solution(var_value_dict=xs_vars)
        self.pfm.add_mip_start(mip_start_sol=mip_start)
