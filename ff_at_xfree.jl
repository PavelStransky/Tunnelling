# Fidelity FF(E0) at a single position x_free = x0 + p0·t/m of the free packet for several packet
# widths, without the full time sweep: exact_twa_overlap.jl run with the parameters below
# (all others are taken from there). See README.md.
# Run:  julia --project -t auto ff_at_xfree.jl

include("exact_twa_overlap.jl")

# ============================================================================
# Parameters (override those of exact_twa_overlap.jl)
# ============================================================================
params = merge(params, (
    x_free     = 60.0,
    sigmas     = (2.0),
    potential  = V2,
    n_energies = 751,
    E_min      = 0.2,
    E_max      = 1.7,
))

if abspath(PROGRAM_FILE) == @__FILE__
    main(params)
end
