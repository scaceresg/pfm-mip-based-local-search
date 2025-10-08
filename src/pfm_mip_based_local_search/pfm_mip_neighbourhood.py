from copy import deepcopy
from time import time

import numpy as np

from .pfm_mip_model import PFMmip


# Class for the MIP-based Neighbourhood Search approach (v2 - 10 May, 2024)
class PFMNeighbourhoodSearch(PFMmip):

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

    # Matheuristic (alg 7): Removing alg_framework, shift_approach, max_tries parameters (all defined at this stage)
    def run_matheuristic(
        self,
        mip_neighb_operator: str,
        param_size: int = 10,
        dynamic_parameter: bool = False,
        param_size_range: list = [5, 10],
        max_tries: int = 10,
        random_seed: int = None,
    ):
        """Returns the makespan value, job sequence and total runtime in a dictionary structure for the MIP-based
            Neighbourhood Search Algorithm for the Permutation Flowshop Scheduling with Makespan criterion (PFM)

        Args:
            mip_neighb_operator (str): The MIP neighbourhood operator to use. Possible alternatives include: 'pos_block',
                                    'gen_swap', 'delta' (deterministic), 'rand_delta' and 'extended'.
            param_size (int, optional): The parameter size value to use in the neighbourhood operator. It's only used for static
                            parameter size, i.e., 'dynamic_parameter' is set to False. Defaults to 10.
            dynamic_parameter (bool, optional): If set to True, the parameter size is increased dynamically based on the
                                    improvement of the current solution and using 'param_size_range' values. Defaults to False.
            param_size_range (list, optional): A list with the minimum and maximum values for the dynamic parameter size:
                                    [min_size, max_size]. Defaults to [5, 10].
            max_tries (int, optional): Maximum tries for MIP neighbourhood operators with static parameter size values. It
                                    Defaults to 10.
            random_seed (int, optional): Random seed to use in the neighbourhood operator. Not applicable to the 'delta'
                                    operator (deterministic). Defaults to None.
        Raises:
            ValueError: if mip_neighb_operator not in {'pos_block', 'gen_swap', 'delta', 'rand_delta', 'extended', 'random'}
            ValueError: if max_size <= min_size

        Returns:
            dict: A dictionary containing the makespan value (key='makespan'), job sequence (key='job_sequence') and total
                runtime (key='runtime')
        """

        ########### Check and Set Initial Values ###########

        # Check mip_neighb_operator value
        if mip_neighb_operator not in {
            "pos_block",
            "gen_swap",
            "delta",
            "rand_delta",
            "extended",
            "random",
        }:
            raise ValueError(
                "Argument 'mip_neighb_operator' should be one of the following: 'pos_block', 'gen_swap',\
                              'delta', 'rand_delta' or 'extended'"
            )
        else:
            self.mip_neighb_operator = mip_neighb_operator

        # Set param_size value
        if not dynamic_parameter:
            self.param_size = param_size

        min_size, max_size = param_size_range
        if max_size <= min_size:
            raise ValueError(
                "'max_size' in 'param_size_range' should be greater than 'min_size'"
            )

        if self.n == 20 and dynamic_parameter:
            if max_size > 9:
                max_size = 9
                print(
                    f"'max_size' value has been adjusted to 9 for the dynamic parameter: num_jobs={self.n}"
                )
            if max_size <= min_size:
                min_size = max_size - 1
                print(
                    f"'min_size' value has been adjusted to {min_size} for the dynamic parameter: num_jobs={self.n}"
                )
        elif self.n == 20 and not dynamic_parameter:
            if self.param_size > 9:
                self.param_size = 9
                print(
                    f"'param_size' value has been adjusted to 9 for the static parameter: num_jobs={self.n}"
                )
        else:
            pass

        # Set random generator
        self.rnd = np.random.default_rng(random_seed)

        # Set a dictionary with the neigbhourhood operator methods
        neighb_operators = {
            "pos_block": self.run_pos_block_neighb,
            "gen_swap": self.run_gen_swap_neighb,
            "delta": self.run_delta_neighb,
            "rand_delta": self.run_rand_delta_neighb,
            "extended": self.run_extended_neighb,
        }

        if not dynamic_parameter:
            print(
                f"Running MIP-based Neighbourhood Search: {self.data_file} <{mip_neighb_operator}-{param_size}> in round {random_seed}"
            )
        else:
            print(
                f"Running MIP-based Neighbourhood Search: {self.data_file} <{mip_neighb_operator}-dynamic-{param_size}> in round {random_seed}"
            )

        ########### Run Main Matheuristic ###########

        # Set timer
        start_timer = time()

        # Get initial solution by NEH
        self.makespan, self.job_sequence = self.run_NEH_algorithm()

        # Build MIP model and set CPLEX parameters by user
        self.build_model(lp_relaxed=False)
        self.set_model_parameters(
            emphasis=self.mip_emphasis,
            mip_solution_limit=self.mip_sol_limit,
            time_limit=self.time_limit,
            n_threads=self.n_threads,
        )

        # Improve initial solution by applying Neighbourhood Search
        improved = True
        while improved:

            improved = False

            # Apply shift local search
            shifted_makespan, shifted_job_seq = self.run_shift_best_improv(
                job_seq=self.job_sequence, seq_makespan=self.makespan
            )

            if shifted_makespan < self.makespan:
                self.makespan = shifted_makespan
                self.job_sequence = shifted_job_seq

                improved = True

            if not improved:

                # Run neighbourhood operator dynamically if dynamic_parameter == True
                # Else: run static neighbourhood operator up to max_tries
                if dynamic_parameter:
                    improved = self.neighb_procedure_dynamic(
                        dict_mip_operators=neighb_operators,
                        min_param_size=min_size,
                        max_param_size=max_size,
                    )
                else:
                    improved = self.neighb_procedure(
                        dict_mip_operators=neighb_operators, max_num_tries=max_tries
                    )

        # Compute total runtime
        matheuristic_runtime = time() - start_timer

        return {
            "makespan": self.makespan,
            "job_sequence": self.job_sequence,
            "runtime": matheuristic_runtime,
        }

    def run_neh_shift_search(self):
        """Execute a hybrid NEH heuristic with shift-based local search improvement.

        Implements a two-phase optimization approach:
        1. Generates an initial solution using the NEH (Nawaz-Enscore-Ham) algorithm
        2. Applies shift-based neighborhood search for local improvement

        The method uses best improvement strategy and updates the instance's makespan
        and job sequence with the best solution found.

        Returns:
            dict: A dictionary containing the results of the search:
                {
                    "makespan": self.makespan,
                    "job_sequence": self.job_sequence,
                    "runtime": ls_runtime,
                }
        """

        ########### Run Local Search ###########

        # Set timer
        start_timer = time()

        # Run NEH
        self.makespan, self.job_sequence = self.run_NEH_algorithm()

        # Improve initial solution by applying Neighbourhood Search
        improved = True
        while improved:

            improved = False

            # Apply shift local search
            shifted_makespan, shifted_job_seq = self.run_shift_best_improv(
                job_seq=self.job_sequence, seq_makespan=self.makespan
            )

            if shifted_makespan < self.makespan:
                self.makespan = shifted_makespan
                self.job_sequence = shifted_job_seq

                improved = True

        # Compute total runtime
        ls_runtime = time() - start_timer

        return {
            "makespan": self.makespan,
            "job_sequence": self.job_sequence,
            "runtime": ls_runtime,
        }

    ############### NEIGHBOURHOOD OPERATOR METHODS ###############

    # Neighbourhood procedure for static param_size using max_tries
    def neighb_procedure(self, dict_mip_operators: dict, max_num_tries: int):
        """Execute neighborhood search procedure with static parameter size.

        Applies the selected MIP-based neighborhood operator for a maximum number
        of iterations, accepting the first improving solution found.

        Args:
            dict_mip_operators (dict): Dictionary mapping operator names to their
                corresponding method functions. Expected keys include 'pos_block',
                'gen_swap', 'delta', 'rand_delta', 'extended'.
            max_num_tries (int): Maximum number of iterations to attempt before
                terminating without improvement. For 'delta' operator, this is
                automatically set to 1.

        Returns:
            bool: True if an improving solution was found and accepted,
                False if no improvement was achieved within max_num_tries.

        Note:
            This method updates self.makespan and self.job_sequence if an
            improvement is found. The 'delta' operator is deterministic and
            only requires one iteration.
        """

        if self.mip_neighb_operator == "delta":
            max_num_tries = 1

        # Run neighbourhood operator in max_tries
        counter = 1
        while counter <= max_num_tries:

            mip_makespan, mip_job_sequence = dict_mip_operators[
                self.mip_neighb_operator
            ]()

            if mip_makespan < self.makespan:
                self.makespan = mip_makespan
                self.job_sequence = mip_job_sequence

                return True
            else:
                counter += 1

        return False

    # Neighbourhood procedure for dynamic param_size
    def neighb_procedure_dynamic(
        self, dict_mip_operators: dict, min_param_size: int, max_param_size: int
    ):
        """Execute neighborhood search with dynamically increasing parameter size.

        Iteratively increases the neighborhood parameter size from minimum to maximum
        until an improving solution is found or the maximum size is reached.

        Args:
            dict_mip_operators (dict): Dictionary mapping operator names to their
                corresponding method functions.
            min_param_size (int): Starting parameter size for the neighborhood operator.
            max_param_size (int): Maximum parameter size to attempt before terminating.

        Returns:
            bool: True if an improving solution was found at any parameter size,
                False if no improvement was achieved across all parameter sizes.

        Note:
            This method modifies self.param_size during execution and updates
            self.makespan and self.job_sequence if an improvement is found.
            The parameter size increases by 1 in each iteration.
        """

        # Run dynamic neighbourhood operator
        self.param_size = min_param_size
        while self.param_size <= max_param_size:

            mip_makespan, mip_job_sequence = dict_mip_operators[
                self.mip_neighb_operator
            ]()

            if mip_makespan < self.makespan:
                self.makespan = mip_makespan
                self.job_sequence = mip_job_sequence

                return True
            else:
                self.param_size += 1

        return False

    ############### NEIGHBOURHOOD OPERATOR METHODS ###############

    # 'pos_block': Position Blocks
    def run_pos_block_neighb(self):
        """Execute position-block neighborhood search operator.

        Implements a MIP-based neighborhood operator that fixes certain job-position
        assignments in blocks and optimizes the remaining assignments. The operator
        selects a random starting position and creates a block of size (2*param_size + 1),
        fixing assignments outside this block.

        Returns:
            tuple: (makespan, job_sequence) where:
                - makespan (int): The makespan value of the best solution found
                - job_sequence (list): The job permutation corresponding to the best solution

        Raises:
            ValueError: If the MIP solution is infeasible or returns a null value.
        """

        # Get incumbent solution
        xs_vars = self.get_xs_from_sequence(seq_sol=self.job_sequence)

        # Set parameters for neighbourhood
        t = self.rnd.integers(1, self.n + 1)
        block = 2 * self.param_size + 1

        # Find block positions
        block_vars = []
        if t + block <= self.n:
            for k in range(1, t):

                j = self.job_sequence[k - 1]
                block_vars.append(self.xs[(j, k)])

            for k in range(t + block, self.n + 1):

                j = self.job_sequence[k - 1]
                block_vars.append(self.xs[(j, k)])
        else:
            for k in range(t + block - self.n, t):

                j = self.job_sequence[k - 1]
                block_vars.append(self.xs[(j, k)])

        # Change LBs for block variables
        self.pfm.change_var_lower_bounds(block_vars, [1] * len(block_vars))

        # Add incumbent solution
        self.add_mip_solution(xs_vars=xs_vars)

        # Solve reduced MIP and get solution
        mip_makespan = self.solve_model(show_log=self.show_output)
        xs_vars = self.get_x_var_vals()
        mip_job_sequence = self.get_sequence_from_mip(xs_vars=xs_vars)

        # Change LBs for block variables
        self.pfm.change_var_lower_bounds(block_vars, [0] * len(block_vars))

        return mip_makespan, mip_job_sequence

    # 'gen_swap': Generalised Swap
    def run_gen_swap_neighb(self):
        """Execute generalized swap neighborhood search operator.

        Implements a MIP-based neighborhood operator that randomly selects job pairs
        for potential swapping while fixing other job-position assignments. The number
        of job pairs considered is determined by self.param_size.

        Returns:
            tuple: (makespan, job_sequence) where:
                - makespan (int): The makespan value of the best solution found
                - job_sequence (list): The job permutation corresponding to the best solution

        Raises:
            ValueError: If the MIP solution is infeasible or returns a null value.
        """

        # Get incumbent solution
        xs_vars = self.get_xs_from_sequence(seq_sol=self.job_sequence)

        # Set parameters for neighbourhood
        set_size = 2 * self.param_size + 1
        moving_pos = self.rnd.choice(self.seq, size=set_size, replace=False)

        # Find swap positions
        fixed_vars = []
        for k in range(1, self.n + 1):

            if k not in moving_pos:

                j = self.job_sequence[k - 1]
                fixed_vars.append(self.xs[(j, k)])

        # Change LBs for swap variables
        self.pfm.change_var_lower_bounds(fixed_vars, [1] * len(fixed_vars))

        # Add incumbent solution
        self.add_mip_solution(xs_vars=xs_vars)

        # Solve reduced MIP and get solution
        mip_makespan = self.solve_model(show_log=self.show_output)
        xs_vars = self.get_x_var_vals()
        mip_job_sequence = self.get_sequence_from_mip(xs_vars=xs_vars)

        # Change LBs for swap variables
        self.pfm.change_var_lower_bounds(fixed_vars, [0] * len(fixed_vars))

        return mip_makespan, mip_job_sequence

    # 'delta': Delta
    def run_delta_neighb(self):
        """Execute deterministic delta neighborhood search operator.

        Implements a systematic MIP-based neighborhood operator that explores
        position shifts for consecutive job segments. The operator considers
        moving blocks of jobs to different positions based on the delta parameter
        (param_size), providing deterministic neighborhood exploration.

        Returns:
            tuple: (makespan, job_sequence) where:
                - makespan (int): The makespan value of the best solution found
                - job_sequence (list): The job permutation corresponding to the best solution

        Raises:
            ValueError: If the MIP solution is infeasible or returns a null value.
        """

        # Get incumbent solution
        xs_vars = self.get_xs_from_sequence(seq_sol=self.job_sequence)

        # Set parameters for neighbourhood
        delta = self.param_size
        job_positions = self.get_job_positions_from_sequence(xs_vars=xs_vars)

        # Find delta positions
        delta_vars = []
        for j in self.jobs:

            j_pos = job_positions[j - 1]

            if j_pos + delta > self.n:
                for k in range(j_pos + delta - self.n + 1, j_pos - delta):

                    delta_vars.append(self.xs[(j, k)])

            elif j_pos - delta <= 0:
                for k in range(j_pos + delta + 1, j_pos - delta + self.n):

                    delta_vars.append(self.xs[(j, k)])
            else:
                for k in range(1, j_pos - delta):

                    delta_vars.append(self.xs[(j, k)])

                for k in range(j_pos + delta + 1, self.n + 1):

                    delta_vars.append(self.xs[(j, k)])

        # Change UBs for delta variables
        self.pfm.change_var_upper_bounds(delta_vars, [0] * len(delta_vars))

        # Add incumbent solution
        self.add_mip_solution(xs_vars=xs_vars)

        # Solve reduced MIP and get solution
        mip_makespan = self.solve_model(show_log=self.show_output)
        xs_vars = self.get_x_var_vals()
        mip_job_sequence = self.get_sequence_from_mip(xs_vars=xs_vars)

        # Change UBs for delta variables
        self.pfm.change_var_upper_bounds(delta_vars, [1] * len(delta_vars))

        return mip_makespan, mip_job_sequence

    # 'rand_delta': Randomised Delta
    def run_rand_delta_neighb(self):
        """Execute randomized delta neighborhood search operator.

        Implements a randomized variant of the delta neighborhood operator that
        randomly selects starting positions and job blocks for position shifting.
        Combines the systematic approach of delta with stochastic exploration.

        Returns:
            tuple: (makespan, job_sequence) where:
                - makespan (int): The makespan value of the best solution found
                - job_sequence (list): The job permutation corresponding to the best solution

        Raises:
            ValueError: If the MIP solution is infeasible or returns a null value.
        """

        # Get incumbent solution
        xs_vars = self.get_xs_from_sequence(seq_sol=self.job_sequence)

        # Set parameters for neighbourhood
        delta = self.param_size
        job_positions = self.get_job_positions_from_sequence(xs_vars=xs_vars)

        # Find random delta positions
        rnd_delta_vars = []
        for j in self.jobs:

            j_pos = job_positions[j - 1]
            rnd_delta = self.rnd.integers(1, 2 * delta)

            if j_pos + rnd_delta > self.n:
                for k in range(j_pos + rnd_delta - self.n + 1, j_pos - rnd_delta):

                    rnd_delta_vars.append(self.xs[(j, k)])

            elif j_pos - rnd_delta <= 0:
                for k in range(j_pos + rnd_delta + 1, j_pos - rnd_delta + self.n):

                    rnd_delta_vars.append(self.xs[(j, k)])

            else:
                for k in range(1, j_pos - rnd_delta):

                    rnd_delta_vars.append(self.xs[(j, k)])

                for k in range(j_pos + rnd_delta + 1, self.n + 1):

                    rnd_delta_vars.append(self.xs[(j, k)])

        # Change UBs for delta variables
        self.pfm.change_var_upper_bounds(rnd_delta_vars, [0] * len(rnd_delta_vars))

        # Add incumbent solution
        self.add_mip_solution(xs_vars=xs_vars)

        # Solve reduced MIP and get solution
        mip_makespan = self.solve_model(show_log=self.show_output)
        xs_vars = self.get_x_var_vals()
        mip_job_sequence = self.get_sequence_from_mip(xs_vars=xs_vars)

        # Change UBs for delta variables
        self.pfm.change_var_upper_bounds(rnd_delta_vars, [1] * len(rnd_delta_vars))

        return mip_makespan, mip_job_sequence

    # 'extended': Extended
    def run_extended_neighb(self):
        """Execute extended neighborhood search operator.

        Implements a comprehensive MIP-based neighborhood operator that combines
        multiple neighborhood structures for broader solution space exploration.
        This operator typically allows more freedom in job reassignments compared
        to other specialized operators.

        Returns:
            tuple: (makespan, job_sequence) where:
                - makespan (int): The makespan value of the best solution found
                - job_sequence (list): The job permutation corresponding to the best solution

        Raises:
            ValueError: If the MIP solution is infeasible or returns a null value.
        """

        # Get incumbent solution
        xs_vars = self.get_xs_from_sequence(seq_sol=self.job_sequence)

        # Set parameters for neighbourhood
        set_size = self.param_size
        job_positions = self.get_job_positions_from_sequence(xs_vars=xs_vars)
        moving_jobs = self.rnd.choice(self.jobs, size=set_size, replace=True)

        # Get T
        moving_pos = []
        for j in moving_jobs:
            moving_pos.append(job_positions[j - 1])

        # Find swap positions
        fixed_vars = []
        for j in self.jobs:
            if j not in moving_jobs:

                j_pos = job_positions[j - 1]

                t_pos = []
                for i in moving_pos:
                    if i < j_pos:
                        t_pos.append(i)

                h = len(t_pos)

                for k in range(1, j_pos - h):

                    fixed_vars.append(self.xs[(j, k)])

                for k in range(j_pos + set_size - h + 1, self.n + 1):

                    fixed_vars.append(self.xs[(j, k)])

        # Change UBs for swap variables
        self.pfm.change_var_upper_bounds(fixed_vars, [0] * len(fixed_vars))

        # Add incumbent solution
        self.add_mip_solution(xs_vars=xs_vars)

        # Solve reduced MIP and get solution
        mip_makespan = self.solve_model(show_log=self.show_output)
        xs_vars = self.get_x_var_vals()
        mip_job_sequence = self.get_sequence_from_mip(xs_vars=xs_vars)

        # Change UBs for swap variables
        self.pfm.change_var_upper_bounds(fixed_vars, [1] * len(fixed_vars))

        return mip_makespan, mip_job_sequence

    ############### CPLEX METHODS ###############

    # Set CPLEX parameter values by the user before building the model
    def set_cplex_params_user(
        self,
        mip_emph: int = 0,
        mip_sol_lim: int = None,
        time_limit: int = None,
        n_threads: int = None,
        show_output: bool = False,
    ):
        """Set CPLEX parameters before building the model.

        Args:
            mip_emph (int, optional): CPLEX MIP Emphasis. Controls trade-offs between speed, feasibility, optimality,
                                    and moving bounds in MIP. Possible alternatives include: 0: BALANCED (Default),
                                    1: FEASIBILITY, 2: OPTIMALITY, 3: BESTBOUND, 4: HIDDENFEAS.
            mip_sol_lim (int, optional): CPLEX MIP Integer Solution Limit. Sets the number of MIP solutions to be found
                                    before stopping. Defaults to None (9223372036800000000 solutions).
            time_limit (int, optional): CPLEX Optimiser Time Limit in Seconds. Sets the maximum time, in seconds, for
                                    a call to an optimizer. Clock type for computation time is set to Wall Clock Time
                                    (total physical time elapsed). Defaults to None.
            n_threads (int, optional): CPLEX Global Thread Count. Sets CPLEX parallel optimisation mode and the default
                                    number of threads that will be invoked. The number of threads is limited by available
                                    processors and Processor Value Units (PVU). Defaults to None (Automatic: let CPLEX
                                    decide).
            show_output (bool, optional): Shows the output log when calling the solve() method. Defaults to False.
        """
        self.mip_emphasis = mip_emph
        self.mip_sol_limit = mip_sol_lim
        self.time_limit = time_limit
        self.n_threads = n_threads
        self.show_output = show_output

    # Get xs variable values from a job sequence
    def get_xs_from_sequence(self, seq_sol: list):

        xs_vars = {}
        for j in self.jobs:
            for k in self.seq:

                if seq_sol[k - 1] == j:
                    xs_vars[f"x_{j}_{k}"] = 1.0
                else:
                    xs_vars[f"x_{j}_{k}"] = 0

        return xs_vars

    # Get job sequence from xs variable values
    def get_sequence_from_mip(self, xs_vars: dict):

        job_seq = [0] * self.n
        for j in self.jobs:
            for k in self.seq:

                if xs_vars[f"x_{j}_{k}"] > 0.998:
                    job_seq[k - 1] = j

        return job_seq

    # Get positions from job sequence
    def get_job_positions_from_sequence(self, xs_vars: dict):

        job_pos = []
        for j in self.jobs:
            t = 1
            while xs_vars[f"x_{j}_{t}"] != 1:
                t += 1

            job_pos.append(t)

        return job_pos

    ############### LOCAL SEARCH APPROACHES ###############

    # Nawaz-Enscore-Ham (NEH) Algorithm (1990)
    def run_NEH_algorithm(self):
        """Execute the NEH (Nawaz-Enscore-Ham) constructive heuristic algorithm.

        Implements the classic NEH algorithm for the Permutation Flow Shop Scheduling
        Problem. The algorithm sorts jobs by decreasing total processing time and
        iteratively inserts each job into the best position of the partial sequence.

        Returns:
            tuple: (makespan, job_sequence) where:
                - makespan (int): The makespan value of the NEH solution
                - job_sequence (list): The job permutation found by NEH algorithm

        Algorithm Steps:
            1. Sort jobs by decreasing sum of processing times across all machines
            2. Initialize sequence with the first job
            3. For each remaining job, insert it at the position that minimizes makespan
        """

        # Sort jobs in non-decreasing order of total processing times
        sum_ptimes = [(self.proc_times[:, j - 1].sum(), j) for j in self.jobs]
        sum_ptimes = sorted(sum_ptimes, reverse=True)

        # Insert first two jobs in the order that minimises the makespan
        initial_seqs = [
            [sum_ptimes[0][1], sum_ptimes[1][1]],
            [sum_ptimes[1][1], sum_ptimes[0][1]],
        ]
        makespans = [self.compute_makespan(par_seq) for par_seq in initial_seqs]
        job_sequence = initial_seqs[
            makespans.index(min(makespans))
        ]  # Future final sequence

        # Insert k = 3 to n jobs
        for sumpt_k in sum_ptimes[2:]:

            k = sumpt_k[1]
            job_sequence = self.job_insertion(job_sequence, k)

        # Compute makespan for the final sequence
        makespan = self.compute_makespan(job_sequence)

        return makespan, job_sequence

    # Shift algorithm ("Best Improvement"): Without restarting search
    def run_shift_best_improv(self, job_seq: list, seq_makespan: int):
        """Execute shift-based best improvement local search.

        Applies a shift neighborhood local search using best improvement strategy.
        For each job in the sequence, tries moving it to all other possible positions
        and selects the move that provides the best makespan improvement.

        Args:
            job_seq (list): Initial job sequence to improve. List of job indices
                representing the current permutation.
            seq_makespan (int): Current makespan value of the input sequence.

        Returns:
            tuple: (best_makespan, best_sequence) where:
                - best_makespan (int): The best makespan found after local search
                - best_sequence (list): The job sequence corresponding to best makespan
        """

        # Save current makespan and sequence
        best_makespan = seq_makespan
        best_sequence = deepcopy(job_seq)

        # Copy current sequence for shifting
        shifted_sequence = deepcopy(job_seq)

        # Shift each job in the sequence
        for k in job_seq:

            shifted_sequence.remove(k)
            shifted_sequence = self.job_insertion(shifted_sequence, k)

            shifted_makespan = self.compute_makespan(shifted_sequence)

            # Update if the new makespan is better
            if shifted_makespan < best_makespan:
                best_makespan = shifted_makespan
                best_sequence = deepcopy(shifted_sequence)
            elif shifted_sequence != best_sequence:
                shifted_sequence = deepcopy(best_sequence)

        return best_makespan, best_sequence

    ############### NEH/SHIFT METHODS ###############

    # Compute makespan
    def compute_makespan(self, job_seq: list):

        completion_machines = [0] * self.m

        for j in job_seq:
            completion_jobs = 0
            for i in self.machines:
                makespan = (
                    max(completion_machines[i - 1], completion_jobs)
                    + self.proc_times[i - 1, j - 1]
                )
                completion_machines[i - 1] = makespan
                completion_jobs = makespan

        return makespan

    # Job insertion method (Taillard, 1990)
    def job_insertion(self, job_seq: list, k_job: int):

        e = self.compute_earliest_completion(job_seq)
        q = self.compute_tail_times(job_seq)
        f = self.compute_earliest_rel_completion(job_seq, k_job, e)
        k_ind = self.find_insertion_pos(job_seq, f, q)

        job_seq.insert(k_ind, k_job)

        return job_seq

    ############### JOB INSERTION METHODS ###############

    # earliest completion times (e_{ij})
    def compute_earliest_completion(self, partial_seq: list):

        e_ij = np.zeros((self.m, len(partial_seq)), dtype=np.int32)

        for j_ind, j in enumerate(partial_seq):
            for i_ind, i in enumerate(self.machines):
                if i_ind == 0 and j_ind == 0:
                    e_ij[i_ind, j_ind] = self.proc_times[i - 1, j - 1]
                elif i_ind == 0:
                    e_ij[i_ind, j_ind] = (
                        e_ij[i_ind, j_ind - 1] + self.proc_times[i - 1, j - 1]
                    )
                elif j_ind == 0:
                    e_ij[i_ind, j_ind] = (
                        e_ij[i_ind - 1, j_ind] + self.proc_times[i - 1, j - 1]
                    )
                else:
                    e_ij[i_ind, j_ind] = (
                        max(e_ij[i_ind - 1, j_ind], e_ij[i_ind, j_ind - 1])
                        + self.proc_times[i - 1, j - 1]
                    )

        return e_ij

    # tail times (q_{ij})
    def compute_tail_times(self, partial_seq: list):

        q_ij = np.zeros((self.m, len(partial_seq)), dtype=np.int32)
        flipped_machs = np.flip(self.machines)
        flipped_part_seq = np.flip(np.array(partial_seq))

        j_ind = len(flipped_part_seq) - 1
        for j in flipped_part_seq:
            i_ind = self.m - 1
            for i in flipped_machs:
                if i_ind == self.m - 1 and j_ind == len(flipped_part_seq) - 1:
                    q_ij[i_ind, j_ind] = self.proc_times[i - 1, j - 1]
                elif i_ind == self.m - 1:
                    q_ij[i_ind, j_ind] = (
                        q_ij[i_ind, j_ind + 1] + self.proc_times[i - 1, j - 1]
                    )
                elif j_ind == len(flipped_part_seq) - 1:
                    q_ij[i_ind, j_ind] = (
                        q_ij[i_ind + 1, j_ind] + self.proc_times[i - 1, j - 1]
                    )
                else:
                    q_ij[i_ind, j_ind] = (
                        max(q_ij[i_ind + 1, j_ind], q_ij[i_ind, j_ind + 1])
                        + self.proc_times[i - 1, j - 1]
                    )
                i_ind -= 1
            j_ind -= 1

        return q_ij

    # earliest relative completion times (f_{ij}) for job k to insert
    def compute_earliest_rel_completion(
        self, partial_seq: list, job_k: int, earliest_compl: np.ndarray
    ):

        f_ij = np.zeros((self.m, len(partial_seq) + 1), dtype=np.int32)

        for j_ind in range(len(partial_seq) + 1):
            for i_ind, i in enumerate(self.machines):
                if i_ind == 0 and j_ind == 0:
                    f_ij[i_ind, j_ind] = self.proc_times[i - 1, job_k - 1]
                elif i_ind == 0:
                    f_ij[i_ind, j_ind] = (
                        earliest_compl[i_ind, j_ind - 1]
                        + self.proc_times[i - 1, job_k - 1]
                    )
                elif j_ind == 0:
                    f_ij[i_ind, j_ind] = (
                        f_ij[i_ind - 1, j_ind] + self.proc_times[i - 1, job_k - 1]
                    )
                else:
                    f_ij[i_ind, j_ind] = (
                        max(f_ij[i_ind - 1, j_ind], earliest_compl[i_ind, j_ind - 1])
                        + self.proc_times[i - 1, job_k - 1]
                    )

        return f_ij

    # Find best job k insertion position
    def find_insertion_pos(
        self, partial_seq: list, earliest_rel_compl: np.ndarray, tail_times: np.ndarray
    ):

        M_list = []

        # Find partial makespan (Mk) for all positions k
        for k_ind in range(len(partial_seq) + 1):  # For index (for k) in partial sequence
            max_M = 0
            for i in self.machines:
                i_ind = i - 1
                if k_ind == len(partial_seq):
                    partial_cmax = earliest_rel_compl[i_ind, k_ind]
                else:
                    partial_cmax = (
                        earliest_rel_compl[i_ind, k_ind] + tail_times[i_ind, k_ind]
                    )
                if partial_cmax > max_M:
                    max_M = partial_cmax
            M_list.append(max_M)

        # Find position with the minimum partial makespan
        k_ind = M_list.index(min(M_list))

        return k_ind
