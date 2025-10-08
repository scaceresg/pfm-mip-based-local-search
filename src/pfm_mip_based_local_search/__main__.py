#!/usr/bin/env python3
"""
Entry point for pfm-mip-based-local-search package.
"""

import argparse
import sys

from . import PFMNeighbourhoodSearch


def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(description="PFM MIP-based Local Search Algorithm")
    parser.add_argument(
        "--instance", required=True, help="Instance file name (e.g., tai20_5_1.txt)"
    )
    parser.add_argument(
        "--inst-type",
        choices=["taillard", "vallada"],
        required=True,
        help="Instance type: taillard or vallada",
    )
    parser.add_argument(
        "--operator",
        choices=["pos_block", "gen_swap", "delta", "rand_delta", "extended", "random"],
        default="pos_block",
        help="MIP neighbourhood operator",
    )
    parser.add_argument(
        "--param-size",
        type=int,
        default=10,
        help="Parameter size for neighbourhood operator",
    )

    args = parser.parse_args()

    try:
        # Initialize the problem
        pfm = PFMNeighbourhoodSearch(data_file=args.instance, inst_name=args.inst_type)

        # Set basic CPLEX parameters
        pfm.set_cplex_params_user(
            mip_emph=4, mip_sol_lim=2, time_limit=900, n_threads=16  # Hidden feasibility
        )

        # Run the matheuristic algorithm
        result = pfm.run_matheuristic(
            mip_neighb_operator=args.operator, param_size=args.param_size, random_seed=42
        )

        print(f"Best makespan found: {result['makespan']}")
        print(f"Runtime: {result['runtime']:.2f} seconds")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
