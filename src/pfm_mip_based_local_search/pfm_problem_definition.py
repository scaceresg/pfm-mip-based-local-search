import os
import sys

import numpy as np
import pandas as pd


# Class for defining the the PFM problem instance
class PFMproblem:

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

        if data_file == None:
            self.n = n
            self.m = m
            self.jobs = np.arange(1, n + 1)
            self.machines = np.arange(1, m + 1)
            self.seq = np.arange(1, n + 1)
            self.proc_times = np.array(proc_times)
            self.best = best
        else:
            self.data_file = data_file

            if inst_name not in {"taillard", "vallada"}:
                raise ValueError("Argument inst_name should be 'taillard' or 'vallada'")
            elif inst_name == "taillard":
                self.get_taillard()
            elif inst_name == "vallada":
                self.get_vallada()

    # Method for reading Taillard data instances
    def get_taillard(self):
        """Load and parse a Taillard benchmark instance for the PFM problem.

        Reads a Taillard format data file containing problem dimensions, processing
        times, and best known solution values.

        File Format Expected:
            Line 1: n m seed best_known_value
            Lines 2 to m+1: Processing times matrix (m×n) where entry (i,j)
                           represents processing time of job j on machine i

        Raises:
            FileNotFoundError: If the data directory or specific instance file
                does not exist at the expected path.
            NotADirectoryError: If the expected directory path is not a directory.
            ValueError: If processing times matrix dimensions don't match (m,n).

        Note:
            This method sets the following instance attributes:
            - self.n: Number of jobs
            - self.m: Number of machines
            - self.best: Best known makespan value from literature
            - self.jobs: Array of job indices [1, 2, ..., n]
            - self.machines: Array of machine indices [1, 2, ..., m]
            - self.seq: Initial sequence [1, 2, ..., n]
            - self.proc_times: Processing times matrix (m×n numpy array)
        """

        # Get Taillard instances file
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        path = os.path.join(base_dir, "data", "taillard_instances")

        # print(f'Trying file {self.data_file} in path {path}')
        try:
            os.chdir(path)
        except FileNotFoundError:
            print(f"Directory: {path} does not exist")
            sys.exit()
        except NotADirectoryError:
            print(f"{path} is not a directory")
            sys.exit()

        with open(self.data_file) as f:
            lines = f.readlines()

        prob_info = []
        line1 = lines[0].split()
        for val in line1:
            prob_info.append(int(val))

        self.n = prob_info[0]
        self.m = prob_info[1]
        self.best = prob_info[3]

        self.jobs = np.arange(1, self.n + 1)
        self.machines = np.arange(1, self.m + 1)
        self.seq = np.arange(1, self.n + 1)

        p_times = []
        for l in lines[1:]:
            ln = [int(x) for x in l.split()]
            p_times.append(ln)

        self.proc_times = np.array(p_times)

        if self.proc_times.shape[0] != self.m or self.proc_times.shape[1] != self.n:
            raise ValueError("Procesing times must have (m, n) shape")

    # Method for reading Vallada et al. data instances
    def get_vallada(self):
        """Load and parse a Vallada et al. benchmark instance for the PFM problem.

        Reads a Vallada format data file containing problem dimensions and processing
        times. Attempts to load best known solution values from a separate bounds file.

        File Format Expected:
            Line 1: n m
            Lines 2 to n+1: For each job, alternating processing and setup times
                           Format: p_1 s_1 p_2 s_2 ... p_m s_m
                           Where p_i is processing time and s_i is setup time on machine i

        Raises:
            FileNotFoundError: If the data directory or specific instance file
                does not exist at the expected path.
            NotADirectoryError: If the expected directory path is not a directory.
            ValueError: If processing times matrix dimensions don't match (m,n).

        Note:
            This method sets the following instance attributes:
            - self.n: Number of jobs
            - self.m: Number of machines
            - self.best: Best known makespan from bounds file, or None if unavailable
            - self.jobs: Array of job indices [1, 2, ..., n]
            - self.machines: Array of machine indices [1, 2, ..., m]
            - self.seq: Initial sequence [1, 2, ..., n]
            - self.proc_times: Processing times matrix (m×n numpy array)

            matrix is transposed to match the expected (m,n) format.
        """

        # Get Vallada instances file
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        path = os.path.join(base_dir, "data", "vallada_etal_instances")

        try:
            os.chdir(path)
        except FileNotFoundError:
            print(f"Directory: {path} does not exist")
            sys.exit()
        except NotADirectoryError:
            print(f"{path} is not a directory")
            sys.exit()

        # Try to load bounds file, use None if not found
        bounds_file = os.path.join(path, "Vallada-bounds.csv")
        vallada_ubs = {}
        if os.path.exists(bounds_file):
            vallada_ubs = pd.read_csv(bounds_file, index_col=0).to_dict()["ub"]

        with open(self.data_file) as f:
            lines = f.readlines()

        prob_info = []
        line1 = lines[0].split()
        for val in line1:
            prob_info.append(int(val))

        self.n = prob_info[0]
        self.m = prob_info[1]
        self.best = vallada_ubs.get(
            self.data_file, None
        )  # Use None if bounds not available

        self.jobs = np.arange(1, self.n + 1)
        self.machines = np.arange(1, self.m + 1)
        self.seq = np.arange(1, self.n + 1)

        p_times = []
        for l in lines[1:]:
            ln = [int(l.split()[i]) for i in range(1, self.m * 2, 2)]
            p_times.append(ln)

        self.proc_times = np.array(p_times).transpose()

        if self.proc_times.shape[0] != self.m or self.proc_times.shape[1] != self.n:
            raise ValueError("Procesing times must have (m, n) shape")
