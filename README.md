# Semiclassical phase-space simulation of quantum tunnelling: Julia code

Julia scripts behind the numerical results of P. Cejnar and P. Stránský, *Semiclassical
Phase-Space Simulation of Quantum Tunneling*. A Gaussian wave packet scatters on a
one-dimensional barrier. Its dynamics is computed exactly and with the truncated Wigner
approximation (TWA) extended by instantaneous tunnelling.

## Files

| file | computes | paper |
|---|---|---|
| [common.jl](common.jl) | shared library: potentials, Gaussian packet, Wigner transform, TWA trajectories with tunnelling, Numerov scattering states, output | all |
| [wigner_origin.jl](wigner_origin.jl) | exact and TWA Wigner functions `Wq`, `Wc` at selected times, energy contours, boost factors `1/P` of the transmitted part | Fig. 2 |
| [wigner_free_origin.jl](wigner_free_origin.jl) | contours `W = c·W_max` of the free packet (contour `c` encloses probability `1 − c`) | Fig. 2 |
| [exact_twa_overlap.jl](exact_twa_overlap.jl) | overlap `F` (`FF` in the code) of the exact and TWA Wigner functions behind the barrier, maps over (`x_free`, `E0`) | Fig. 3 |
| [ff_at_xfree.jl](ff_at_xfree.jl) | `F(E0)` at a single position `x_free` of the free packet | Fig. 3 |
| [eisenbud_wigner.jl](eisenbud_wigner.jl) | delay `τ(E0)` of the transmitted packet, exact and TWA; Eisenbud–Wigner–Smith time `ħ dφ/dE`; transmission probabilities | Fig. 4 |

## Installation

Julia 1.10 or newer is required. Run once in the repository root:

```
julia setup.jl
```

It installs the packages listed in [Project.toml](Project.toml) (FFTW, Plots, ProgressMeter and the
standard libraries DelimitedFiles, LinearAlgebra, Printf, Random, Statistics) into the environment
of the repository and precompiles them.

## Running

```
julia --project -t auto <script>.jl        # from the repository root
```

Each script defines a named tuple `params` and an entry point `main(par)`, which runs when the
file is executed as a script. To change parameters without editing the file, start
`julia --project -t auto` in the repository root and run:

```julia
include("eisenbud_wigner.jl")
main(merge(params, (; potential = V2, sigmas = [1.0, 2.0, 3.0])))
```

Real-valued parameters must be floats (`m = 100.0`, not `m = 100`). Energy sweeps run one energy
per thread. Each run writes a new folder into the current directory. The folder name gives the
computation and its main parameters: `eisenbud_wigner_…`, `exact_twa_overlap_…` (with the suffix
`_xfree=…` for `ff_at_xfree.jl`), `origin_…` for `wigner_origin.jl` and `origin_free_…` for
`wigner_free_origin.jl`. Text files are tab-separated. `eisenbud_wigner.jl`, `exact_twa_overlap.jl`
and `ff_at_xfree.jl` also save PNG figures.

## Parameters of the paper

| quantity | value | parameter |
|---|---|---|
| units | ħ = 1, m = 100 | `ħ`, `m` |
| potentials | V₁ = exp(−x²), V₂ = (1 + x²) exp(−x²) | `potential = V1`, `V2` |
| initial position | x₀ = −30 | `x0_init` |
| widths | σx = 0.71, 1.41, 2.12 | `sigma0` / `sigmas` = 1.0, 2.0, 3.0 |
| Fig. 2 | E₀ = 0.4; σx = 0.71 and 2.12 | `E0`, `sigma0` |
| Figs. 3, 4 | evaluated when the free packet is at x = 60 | `x_free`, `x_final` = 60.0 |
| Fig. 3 | region behind the barrier, x > 30 | `x_split` = 30.0 |

The equivalent choice ħ = 0.1, m = 1 gives the same results with all times divided by 10.
The potential catalogue in `common.jl` also contains potentials not used in the paper.
Some defaults in the scripts differ from the paper. For example, `exact_twa_overlap.jl` runs
`sigmas = (0.5, 1.0, 3.0)`. Use the values in this table to reproduce a figure.

## Conventions

- **Initial packet:** `ψ₀(x) = (πσ₀²)^(−1/4) exp[−(x − x₀)²/(2σ₀²) + i p₀(x − x₀)/ħ]`.
  The paper's width σx is the standard deviation of `|ψ₀|²`: **σx = σ₀/√2**, σp = ħ/(2σx).
- **Energy:** `E₀ = p₀²/2m + V(x₀)`.
- **Wigner function:** `W(x, p) = (1/πħ) ∫ ψ*(x + y) ψ(x − y) e^{2ipy/ħ} dy`, Eq. (1) of the paper.
- **Transmission amplitude:** `e^{ipx/ħ}` incident from the left gives `t(p) e^{ipx/ħ}` on the
  right (the paper's `a(p̄)`); `t ≡ 1` for `V = 0`.
- **Exact dynamics:** superposition of stationary scattering states obtained by Numerov
  integration (`quantum_method = :stationary`). In `wigner_origin.jl` and `eisenbud_wigner.jl`,
  split-operator propagation (`:time_dependent`) is available as a check. `exact_twa_overlap.jl`
  always uses the stationary states.
- **TWA:** trajectories sampled from `W₀` follow Newton's equations (velocity Verlet). At a turning
  point inside a barrier a trajectory splits. The reflected branch keeps the weight `w(1 − T)`; the
  transmitted branch restarts at the far turning point with the weight `wT`,
  `T = exp[−(2/ħ) ∫ √(2m(V − E)) dx]`. For V₁ and V₂ each trajectory splits once.
- **Delay (Fig. 4):** the `τ = t₁ − t₂` that maximises the overlap `∫∫ W(t₁) W_free(t₂) dx dp` of the
  scattered state at `t₁` with the free packet at `t₂` (for pure states
  `|⟨ψ_free(t₂)|ψ(t₁)⟩|²/2πħ`). It is evaluated when the free packet is at `x_final`. The TWA delay
  uses the same overlap with the transmitted TWA branches (`x > x_R`, `p > 0`).
- **Overlap F (Fig. 3):** `F = ∫∫ Wq Wc dx dp / (‖Wq‖ ‖Wc‖)`, `‖W‖² = ∫∫ W² dx dp`. The exact `Wq`
  (from stationary states) is evaluated at the TWA trajectory points. The region behind the
  barrier is selected by a smooth window `f(x)` that rises from 0 to 1 over `x_split ± ℓ_split`:
  ψ is multiplied by `√f` and the TWA weights by `f`.

## Numerical notes

- The split-operator method has a noise floor of ~1e-13 in ψ. Transmitted probabilities below
  ~1e-20 (V₂ at low E₀) need the stationary method.
- For narrow packets the transmitted weight comes from the far momentum tail of `W₀`; keep
  importance sampling on (`is_fraction > 0`).
- Trajectories are reproducible for a given `rng_seed` and number of threads.
