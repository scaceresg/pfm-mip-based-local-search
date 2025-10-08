# PFM MIP-based Local Search

Permutation Flowshop Scheduling Problem with Makespan criterion (PFM) solver using a MIP-based neighbourhood search (matheuristic). It builds a compact MIP model in DOcplex/CPLEX and explores tailored neighbourhoods (position blocks, generalized swaps, delta variants, and an extended neighbourhood). A classic NEH heuristic is used to seed the search, with optional shift-based local improvement.

## Features
- Compact PFM MIP model (assignment + positional and machine completion constraints)
- Multiple MIP neighbourhood operators:
	- pos_block: position blocks with fixed variables outside the block
	- gen_swap: randomized generalized job swaps under MIP
	- delta: deterministic delta-based shifts
	- rand_delta: randomized delta variant
	- extended: broader neighbourhood relaxing more assignments
- NEH constructive heuristic and shift best-improvement local search
- CLI interface and Makefile workflow

## Requirements
- Python 3.10+
- IBM CPLEX Optimization Studio with DOcplex (docplex>=2.23.0)
- Numpy, Pandas

Data files are expected under:
- `data/taillard_instances/` (Taillard)
- `data/vallada_etal_instances/` (Vallada), optional `Vallada-bounds.csv` with column `ub`

## Quick start

Clone and run from source (Windows PowerShell shown; similar on macOS/Linux):

```powershell
# Get the code
git clone https://github.com/scaceresg/pfm-mip-based-local-search.git
cd pfm-mip-based-local-search

# Create venv and install dev deps (black, isort, types-setuptools)
make dev-setup

# Format imports and code (optional)
make fmt

# Run a Taillard instance with a position-block operator
make run

# Or run the CLI directly (choose operator and parameters)
venv\Scripts\python.exe -m pfm-mip-based-local-search ^
	--instance tai20_5_1.txt ^
	--inst-type taillard ^
	--operator pos_block ^
	--param-size 10
```

On macOS/Linux, replace the last command with:

```bash
venv/bin/python -m pfm-mip-based-local-search \
	--instance tai20_5_1.txt \
	--inst-type taillard \
	--operator pos_block \
	--param-size 10
```

## CLI usage

The package exposes a simple CLI:

```text
--instance    Instance file name (e.g., tai20_5_1.txt) [required]
--inst-type   {taillard, vallada} [required]
--operator    {pos_block, gen_swap, delta, rand_delta, extended, random} (default: pos_block)
--param-size  Integer parameter that controls neighbourhood size (default: 10)
```

Behind the scenes, the entrypoint does the following:
- Loads the instance using `PFMproblem.get_taillard()` or `PFMproblem.get_vallada()`
- Builds the MIP model and applies neighbourhood constraints per operator
- Solves the reduced MIP and extracts the best sequence and makespan

Key algorithmic building blocks (from docstrings):
- `run_NEH_algorithm`: sorts jobs by decreasing total processing time and inserts each job optimally
- `run_shift_best_improv`: shift (insertion) neighbourhood with best-improvement strategy until no improvement
- Neighbourhood operators:
	- `run_pos_block_neighb`: fixes assignments outside a random block of size 2·param_size+1
	- `run_gen_swap_neighb`: samples job pairs for potential swapping
	- `run_delta_neighb`: deterministic block shifts based on `param_size`
	- `run_rand_delta_neighb`: randomized delta exploration
	- `run_extended_neighb`: broader relaxations for wider search

## Makefile highlights

Common targets (cross‑platform):

- `make venv` — Create a virtual environment
- `make install` — Install package in editable mode
- `make install-dev` — Install package + dev tools (black, isort, types)
- `make fmt` — Format code and imports
- `make run` — Run example on Taillard instance (pos_block)
- `make run-vallada` — Run example on a Vallada instance (delta)
- `make build` — Build distributable packages
- `make clean` — Remove build artifacts
- `make clean-all` — Deep clean including venv

Use `make` with no arguments to see the categorized help.

## Programmatic usage (example)

```python
from pfm_mip_based_local_search import PFMNeighbourhoodSearch

pfm = PFMNeighbourhoodSearch(data_file="tai20_5_1.txt", inst_name="taillard")
# Optional: user parameters for CPLEX
pfm.set_cplex_params_user(mip_emph=4, mip_sol_lim=2, time_limit=900, n_threads=16)

# Run the matheuristic
result = pfm.run_matheuristic(mip_neighb_operator="pos_block", param_size=10, random_seed=42)
print(result["makespan"], result["job_sequence"], result["runtime"]) 
```

## Data notes
- Taillard loader (`get_taillard`) expects an m×n processing time matrix after the header; dimensions are validated.
- Vallada loader (`get_vallada`) reads alternating p/s values per machine and ignores setup times; if `Vallada-bounds.csv` is present, `best` is populated from the `ub` column.

## Troubleshooting
- Ensure CPLEX and DOcplex are installed and licensed; the solver is required at runtime.
- Paths: instances must exist under `data/taillard_instances` or `data/vallada_etal_instances`.
- Windows Python: the Makefile uses `python -m venv`; ensure `python` is in PATH.

## License
MIT