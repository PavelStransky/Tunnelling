# Contours of the Wigner function of the free wave packet (V = 0), to be drawn in Origin over the
# heat maps of wigner_origin.jl. See README.md.
# Run:  julia --project wigner_free_origin.jl

include("common.jl")

# ============================================================================
# Parameters (ħ, m, the packet and the times should match wigner_origin.jl)
# ============================================================================
params = (
    ħ = 1.0,
    m = 100.0,

    # initial Gaussian packet
    E0      = 0.4,
    x0_init = -30.0,
    sigma0  = 3.0,

    t_export = [240.0, 330.0, 500.0],

    # contour levels W = c·W_max, c = contour_min + k·contour_step, k = 0 … n_contours − 1 (0 < c < 1);
    # the contour c encloses the probability 1 − c
    contour_min  = 0.02,
    contour_step = 0.58,
    n_contours   = 2,
    n_theta      = 1001,        # points per contour
)

# ============================================================================
# Code
# ============================================================================

# Contour W_free(x, p, t) = c/(πħ): the sheared ellipse u² + v² = −ln c,
# u = (x − x0 − p t/m)/σ0, v = σ0(p − p0)/ħ, as a closed polyline
function free_wigner_contour(c::Float64, t::Float64, x0::Float64, p0::Float64, σ0::Float64,
                             m::Float64, ħ::Float64; n_θ::Int = 1001)
    r = sqrt(-log(c))
    θ = range(0, 2π; length = n_θ)
    p = [p0 + ħ * r * sin(θk) / σ0 for θk in θ]
    x = [x0 + p[k] * t / m + σ0 * r * cos(θ[k]) for k in eachindex(θ)]
    x[end] = x[1]
    p[end] = p[1]
    return x, p
end

function main(par)
    (; ħ, m, E0, x0_init, sigma0, t_export) = par
    out_folder = "origin_free_E=$(E0)_sigma=$(sigma0)"

    p0     = packet_momentum(E0, V0, x0_init, m)
    W_max  = 1 / (π * ħ)
    levels = [par.contour_min + k * par.contour_step for k in 0:par.n_contours-1]
    all(c -> 0 < c < 1, levels) || error("The contour levels c = $(levels) must lie in (0, 1).")
    mkpath(out_folder)

    println("=== Contours of the Wigner function of the free packet for Origin ===")
    @printf("  ħ = %g, m = %g, E0 = %g, x0 = %g, σ0 = %g  →  p0 = %.6g,  W_max = %.6g\n",
            ħ, m, E0, x0_init, sigma0, p0, W_max)
    println("\n  level c    W = c·W_max   enclosed probability")
    for c in levels
        @printf("  %7.3g   %12.6g   %6.3g\n", c, c * W_max, 1 - c)
    end

    labels = [@sprintf("W=%gWmax", c) for c in levels]
    @printf("\n       t   centre x   σ_x(t)   extent of the outer contour (c = %g)\n", levels[1])
    for t in t_export
        header = String[]
        cols   = Vector{Float64}[]
        for (label, c) in zip(labels, levels)
            xs, ps = free_wigner_contour(c, t, x0_init, p0, sigma0, m, ħ; n_θ = par.n_theta)
            err = maximum(abs(free_wigner(xs[k], ps[k], t, x0_init, p0, sigma0, m, ħ) / W_max - c)
                          for k in eachindex(xs))
            err < 1e-10 || error("The contour c = $c at t = $t does not lie on W_free (error $err).")
            push!(header, "x($label)", "p($label)")
            push!(cols, xs, ps)
        end
        write_columns(joinpath(out_folder, @sprintf("Wfree_contours_t=%05.2f.txt", t)), header, cols)
        @printf("  %6.1f  %8.4g  %7.4g   x ∈ [%.4g, %.4g],  p ∈ [%.4g, %.4g]\n",
                t, x0_init + p0 * t / m, free_width(sigma0, t, m, ħ) / sqrt(2),
                minimum(cols[1]), maximum(cols[1]), minimum(cols[2]), maximum(cols[2]))
    end
    println("\nDone. Data saved in: ", out_folder)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(params)
end
