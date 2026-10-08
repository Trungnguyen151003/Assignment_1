# Assignment 2 experiment pipeline

This implementation compares fixed-strength Gaussian mutation with one-global-
step-size self-adaptive Gaussian mutation. Both evolutionary variants use the
same John-set gecko, 16-8-6 neural controller, initial populations, tournament
selection, `(mu + lambda)` survivor selection, simulation settings, fitness,
and evaluation budget. Random search is an equal-budget baseline.

## Controller and genotype

The controller receives six normalized joint angles, six normalized joint
velocities, the planar target vector expressed in the body frame, and a sine/
cosine clock pair. Its 190 weights and biases are stored with `log_sigma` in a
JSON-compatible genotype. Direct tanh outputs are scaled to the hinge range.

The adaptive condition applies

```text
log(sigma') = clip(log(sigma) + tau * N(0, 1))
w_i'        = clip(w_i + sigma' * N_i(0, 1))
```

where `tau = 1 / sqrt(2n)`. The fixed condition draws the same types of random
values but always uses the configured fixed sigma.

## Running the code

Run all commands below from the repository root. The examples use **Git Bash**.
In Git Bash, `\` continues a command onto the next line and must be the final
character on that line: do not put a space after it and do not type the `$`
prompt symbol.

The original `A2_template_2026.py` is retained as the provided demonstration
and reference file. The experiment entry point is `a2_run_experiments`.

### 1. Smoke test

Run this small test first to validate the complete pipeline. Its results must
not be used in the report.

```bash
uv run python -m assignments.assignment_2.a2_run_experiments \
  --variants fixed adaptive random \
  --seeds 999 \
  --population-size 4 \
  --offspring-count 4 \
  --generations 2 \
  --tournament-size 3 \
  --duration 0.2 \
  --fixed-sigma 0.1 \
  --output assignments/assignment_2/results_smoke \
  --quiet
```

This runs three variants with one seed and 12 evaluations per variant. The
variants are executed sequentially, not concurrently.

### 2. Pilot experiment

Pilot seeds are used to verify runtime and lock the experimental parameters.
They must not be reused for final statistical inference.

```bash
uv run python -m assignments.assignment_2.a2_run_experiments \
  --variants fixed adaptive random \
  --seeds 100 101 102 \
  --population-size 20 \
  --offspring-count 20 \
  --generations 30 \
  --tournament-size 3 \
  --duration 5 \
  --fixed-sigma 0.1 \
  --output assignments/assignment_2/results_pilot \
  --quiet
```

Analyse the pilot results with:

```bash
uv run python -m assignments.assignment_2.a2_analyse \
  --results assignments/assignment_2/results_pilot \
  --output assignments/assignment_2/results_pilot/analysis
```

### 3. Final experiment

After fixing all parameters, run the final experiment with ten independent
seeds that were not used in the pilot:

```bash
uv run python -m assignments.assignment_2.a2_run_experiments \
  --variants fixed adaptive random \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
  --population-size 20 \
  --offspring-count 20 \
  --generations 80 \
  --tournament-size 3 \
  --duration 5 \
  --fixed-sigma 0.1 \
  --output assignments/assignment_2/results_final \
  --quiet
```

The evaluation budget for one run is
`population_size + generations * offspring_count`, which is
`20 + 30 * 20 = 620` evaluations here. The complete final experiment performs
`3 variants * 10 seeds * 620 = 18,600` simulations and may take a long time.

Analyse the final results with:

```bash
uv run python -m assignments.assignment_2.a2_analyse \
  --results assignments/assignment_2/results_final \
  --output assignments/assignment_2/results_final/analysis
```

### Running one variant only

For debugging, `--variants` can contain just one condition, for example:

```bash
uv run python -m assignments.assignment_2.a2_run_experiments \
  --variants fixed \
  --seeds 999 \
  --population-size 4 \
  --offspring-count 4 \
  --generations 2 \
  --duration 0.2 \
  --fixed-sigma 0.1 \
  --output assignments/assignment_2/results_fixed_debug \
  --quiet
```

For the final comparison, run all variants with identical parameters and
evaluation budgets. `--fixed-sigma` is the constant mutation strength for the
fixed variant and the initial mutation strength for the adaptive variant; the
random baseline does not mutate weights.

### Output-directory safety

The runner refuses to overwrite an existing run database. If an output folder
already contains a run with the same variant and seed, choose a new output
name, such as `results_smoke_2`. Do not analyse a run that was interrupted
before all requested variants and seeds completed.

## Analysis outputs

The analysis produces:

- `final_results.csv` with run-level final fitness and normalized AUC;
- `statistical_analysis.json` with descriptive statistics, paired bootstrap
  confidence intervals, and paired sign-flip permutation tests;
- convergence, final-performance, and adaptive-sigma figures.

Final target distance is the primary outcome. Normalized best-so-far AUC is the
predefined convergence-speed outcome. Generations and individuals are not
treated as independent statistical observations; the independent unit is one
seeded run.
