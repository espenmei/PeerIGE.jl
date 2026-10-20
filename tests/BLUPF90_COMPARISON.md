# BLUPF90 comparison

`compare_blupf90.jl` is an opt-in integration benchmark; it is intentionally
not included by `tests/runtests.jl`. It compares PeerIGE's direct/social
AI-REML variance estimates and elapsed time with BLUPF90+ on the same observed
data, roster, and sparse `Ainv`.

The data passed to BLUPF90 contains the focal animal ID and one column per
possible peer. The zero-level effects in the parameter file accumulate those
peer columns into one social animal effect, excluding the focal animal. The
direct and social effects share the same pedigree covariance and a 2-by-2
genetic covariance block. This is the social-interaction parameterization
shown in the [BLUPF90 tutorial](https://masuday.github.io/blupf90_tutorial/mrode_c09ex091_social_interaction.html).
The script first runs a six-animal, fixed-variance check and compares all
fixed, direct, and social solutions between BLUPF90+ and PeerIGE. If that
check fails, the REML benchmark does not proceed.

## Run

From the repository root in PowerShell:

```powershell
$env:BLUPF90_EXE = 'C:\Users\espenmei\software\blupf90+.exe'
julia --project=. tests/compare_blupf90.jl
```

The default is a single-thread baseline. For a multi-threaded comparison, give
both programs the same thread count (the script errors if they differ):

```powershell
$env:BLUPF90_THREADS = '8'
julia --threads=8 --project=. tests/compare_blupf90.jl
```

Each run overwrites `compare.csv`, so copy it aside between runs.

The executable can also be found on `PATH`; by default, Windows runs look for
the path above and other platforms look for `blupf90+`. BLUPF90+ must support
the standard `user_file` covariance type. R and `tidyverse` are only needed
if the requested simulation dataset is missing.

The first run writes data and logs under the git-ignored
`outputs/blupf90_compare/`. PeerIGE runs before BLUPF90+, and `compare.csv`
is saved after each program finishes. If you stop a long BLUPF90+ run, the
PeerIGE results are kept. The file contains the simulation-truth
row, one PeerIGE row per Hutchinson seed, and one BLUPF90+ row, including
variance components, timing, thread count, iteration count, and a convergence
flag. The tiny fixed-variance run directory retains the generated
BLUPF90 input and `solutions` output for inspection.

Both REML fits start at `(σ²_d, σ²_s, σ_ds, σ²_e) = (1.0, 0.2, 0.05, 2.0)`.
The small BLUPF90 trial remained at a zero direct/social covariance when
started exactly at zero, so the shared, positive-definite interior start avoids
that boundary for both programs.

## Options

Set environment variables before running:

| Variable | Default | Purpose |
|---|---:|---|
| `BLUPF90_EXE` | Windows: `C:\Users\espenmei\software\blupf90+.exe`; otherwise `blupf90+` | Executable path or command |
| `BLUPF90_DATA_DIR` | `outputs/blupf90_compare/data_ng<ng>` | Directory with the five parameter-recovery CSV inputs; simulated if missing |
| `BLUPF90_NG` | `3750` | Number of groups (`N = 4 * ng`, so 15,000 animals by default) |
| `BLUPF90_MISSING_FRAC` | `0.10` | Missing phenotype fraction when generating data |
| `BLUPF90_PEERIGE_REPEATS` | `3` | PeerIGE runs with distinct Hutchinson seeds |
| `BLUPF90_PEERIGE_K` | `20` | Hutchinson probes per PeerIGE run |
| `BLUPF90_PEERIGE_MAX_ITER` | `15` | AI-REML iteration limit |
| `BLUPF90_MAX_ROUNDS` | `100` | BLUPF90+ VCE iteration limit |
| `BLUPF90_SIGMA_DS_INIT` | `0.05` | Shared nonzero initial direct/social covariance for both programs |
| `BLUPF90_THREADS` | `1` | Thread count set for BLUPF90 and the Julia command |
| `BLUPF90_VALIDATE_ONLY` | `0` | Set to `1` to run only the tiny fixed-variance equivalence check |
| `BLUPF90_NG_LIST` | empty | Optional comma-separated extra simulation sizes, e.g. `25000,100000` |

Extra sizes are generated into separate ignored output folders and appended
to `compare.csv`. At N=20,000, BLUPF90+'s sparse factor reached about
162 million nonzeros and one REML round took roughly an hour on one thread.
Large BLUPF90 jobs can require
substantial RAM and runtime; increase sizes gradually and monitor the
machine. Timing is wall-clock and includes BLUPF90 process startup; Julia's
first compilation is not separately warmed up, so treat the first PeerIGE
timing as a cold-start result.

The estimates need not match exactly: PeerIGE's traces use stochastic
Hutchinson probes, while BLUPF90's REML algorithm and stopping rules differ.
Interpret seed-to-seed spread alongside the estimates, iteration counts, and
times rather than treating this as a bitwise regression test.
