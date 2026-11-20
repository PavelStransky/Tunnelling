# Asymptotic time delay of a Gaussian wave packet as a function of its energy: exact (stationary
# scattering states or split-operator) against semiclassical (TWA with Gamow tunnelling), and the
# stationary Eisenbud–Wigner time ħ dφ/dE. See README.md.
# Run:  julia --project -t auto eisenbud_wigner.jl

using Plots
include("common.jl")

# ============================================================================
# Parameters
# ============================================================================
params = (
    ħ = 1.0,
    m = 100.0,
    potential = V2,

    # initial Gaussian packet; one run (and output folder) per width
    x0_init = -30.0,
    sigmas  = (2.0),

    # energy sweep
    n_energies = 301,
    E_min      = 0.1,
    E_max      = 1.6,

    # exact calculation: :stationary (Numerov) or :time_dependent (split-operator)
    quantum_method = :stationary,

    # stationary calculation; V must be negligible outside [x_L, x_R].
    # Trajectories count as transmitted at x > x_R.
    x_L = -15.0,
    x_R =  15.0,
    numerov_per_λ = 100,        # Numerov step h = λ_min/numerov_per_λ
    dp_per_σp     = 20,         # momentum step ħ/(σ0·dp_per_σp), refined if needed
    n_σp          = 12.0,       # momentum window p0 ± n_σp·ħ/σ0

    # end of the propagation: the time at which the free packet is at x_final
    x_final = 60.0,

    # grid of ψ(x) and absorbing potential (:time_dependent only)
    x_min = -150.0,
    x_max =  150.0,
    x_num = 65536,
    abs_width    = 15.0,
    abs_strength = 0.8,

    # time step of the trajectories and of the split-operator (times scale as √m)
    dt = 0.1,

    # trajectories
    N_traj              = 1_000_000,  # divisible by 4
    rng_seed            = 42,       # the same for all energies
    is_fraction         = 0.5,      # fraction of momenta drawn from the importance proposal (0 = off)
    max_gen             = 50,
    w_min_spawn         = 0.0,
    V_barrier_threshold = 0.05,

    # search for the maximum of the overlap over τ = t1 − t2
    τ_min  = -200.0,
    τ_max  =  400.0,
    τ_step =   10.0,

    # figures
    figure_size = (1200, 700),
    figure_dpi  = 150,
)

# ============================================================================
# Code
# ============================================================================

# O_q(τ) = |Σ_k c_k e^{−iω_k τ}|²/(2πħ)
@inline function quantum_overlap(τ::Float64, c::Vector{ComplexF64}, ω::Vector{Float64}, inv_2πħ::Float64)
    re = 0.0
    im = 0.0
    @inbounds for k in eachindex(c)
        s, co = sincos(ω[k] * τ)
        re += real(c[k]) * co + imag(c[k]) * s
        im += imag(c[k]) * co - real(c[k]) * s
    end
    return (re * re + im * im) * inv_2πħ
end

# O_c(τ) = Σ_k amp_k exp(−(ξ_k + v_k τ)²/σ0²)
@inline function classical_overlap(τ::Float64, amp::Vector{Float64}, ξ::Vector{Float64},
                                   v::Vector{Float64}, inv_σ2::Float64)
    s = 0.0
    @inbounds @simd for k in eachindex(amp)
        d = ξ[k] + v[k] * τ
        s += amp[k] * exp(-d * d * inv_σ2)
    end
    return s
end

# Maximum of f(τ): best point of τ_grid refined by golden-section search between its neighbours.
# Returns (NaN, f_best) if the maximum lies at the edge of the grid.
function find_max_tau(f, τ_grid::AbstractRange{Float64}; tol::Float64 = 1e-8)
    best_i = 1
    best_f = f(τ_grid[1])
    for i in 2:length(τ_grid)
        v = f(τ_grid[i])
        if v > best_f
            best_f = v
            best_i = i
        end
    end
    (best_i == 1 || best_i == length(τ_grid)) && return NaN, best_f

    a = τ_grid[best_i - 1]
    b = τ_grid[best_i + 1]
    g = (sqrt(5.0) - 1) / 2
    c = b - g * (b - a); fc = f(c)
    d = a + g * (b - a); fd = f(d)
    while b - a > tol
        if fc > fd
            b = d; d = c; fd = fc
            c = b - g * (b - a); fc = f(c)
        else
            a = c; c = d; fc = fd
            d = a + g * (b - a); fd = f(d)
        end
    end
    τ = 0.5 * (a + b)
    return τ, f(τ)
end

# Quadrature nodes and weights over p: the grid p_st with the weight dp; ±n_patch steps around
# every narrow resonance are replaced by a dense trapezoidal patch (step Γ_p/20 up to 2Γ_p, then
# growing geometrically). Returns (p, w, t(p)).
function resonance_quadrature(p_st::AbstractRange{Float64}, t_st::Vector{ComplexF64},
                              resonances::Vector{NTuple{2, Float64}}, V::Vector{Float64},
                              x_L::Float64, h::Float64, m::Float64, ħ::Float64;
                              n_patch::Int = 10, patch_ratio::Float64 = 1.01)
    n  = length(p_st)
    dp = step(p_st)
    isempty(resonances) && return collect(p_st), fill(dp, n), copy(t_st)

    # index intervals of p_st around the resonances, overlapping ones merged
    intervals = Tuple{Int, Int, Vector{NTuple{2, Float64}}}[]
    for r in sort(resonances)
        i_r = clamp(floor(Int, (r[1] - first(p_st)) / dp) + 1, 1, n)
        i_a = max(1, i_r - n_patch)
        i_b = min(n, i_r + 1 + n_patch)
        if !isempty(intervals) && i_a ≤ intervals[end][2]
            push!(intervals[end][3], r)
            intervals[end] = (intervals[end][1], max(i_b, intervals[end][2]), intervals[end][3])
        else
            push!(intervals, (i_a, i_b, [r]))
        end
    end

    keep    = trues(n)
    p_patch = Float64[]
    w_patch = Float64[]
    for (i_a, i_b, rs) in intervals
        p_a = p_st[i_a]
        p_b = p_st[i_b]
        pts = [p_a, p_b]
        for (p_r, Γ_p) in rs
            for s in Γ_p .* (0:0.05:2)
                push!(pts, p_r - s, p_r + s)
            end
            s = 2 * Γ_p
            while s < p_b - p_a
                s *= patch_ratio
                push!(pts, p_r - s, p_r + s)
            end
        end
        filter!(p -> p_a ≤ p ≤ p_b, pts)
        sort!(pts)
        unique!(pts)

        w = zeros(length(pts))
        for k in 1:length(pts)-1
            d = pts[k + 1] - pts[k]
            w[k]     += d / 2
            w[k + 1] += d / 2
        end
        w[1]   += dp / 2
        w[end] += dp / 2

        keep[i_a:i_b] .= false
        append!(p_patch, pts)
        append!(w_patch, w)
    end

    t_patch = Vector{ComplexF64}(undef, length(p_patch))
    @threads for k in eachindex(p_patch)
        t_patch[k] = transmission_amplitude(p_patch[k], V, x_L, h, m, ħ)
    end

    p_q  = vcat(p_st[keep], p_patch)
    w_q  = vcat(fill(dp, count(keep)), w_patch)
    t_q  = vcat(t_st[keep], t_patch)
    perm = sortperm(p_q)
    return p_q[perm], w_q[perm], t_q[perm]
end

# t(p) on the momentum quadrature common to all energies (stationary method)
function stationary_setup(par, sigma0::Float64, p0_values::Vector{Float64}, V_top::Float64)
    (; ħ, m, potential, x_L, x_R, n_σp, τ_min, τ_max) = par
    σ_p  = ħ / sigma0
    p_hi = p0_values[end] + n_σp * σ_p
    # the copies of the maximum of O_q, shifted by m·(2πħ/dp)/p, must lie outside [τ_min, τ_max]
    dp_alias = 2π * ħ * m / (1.2 * p_hi * (τ_max - τ_min))
    dp       = min(σ_p / par.dp_per_σp, dp_alias)
    dp == dp_alias && println("  momentum step refined to dp = ", dp)
    p_lo = max(p0_values[1] - n_σp * σ_p, dp)
    p_st = p_lo:dp:p_hi
    p_lo == dp && println("  momentum window cut from below at p = ", p_lo)

    λ_min = 2π * ħ / last(p_st)
    n_x   = ceil(Int, (x_R - x_L) * par.numerov_per_λ / λ_min) + 1
    h     = (x_R - x_L) / (n_x - 1)
    V     = [potential(x_L + (j - 1) * h) for j in 1:n_x]
    V_edge = max(abs(V[1]), abs(V[end]))
    V_edge > 1e-12 * V_top && println("  WARNING: V at the edge of [x_L, x_R] is ", V_edge, " – widen x_L, x_R")
    println("  stationary:   ", length(p_st), " momenta × ", n_x, " Numerov points")

    t_st = Vector{ComplexF64}(undef, length(p_st))
    @time @threads for i in eachindex(p_st)
        t_st[i] = transmission_amplitude(p_st[i], V, x_L, h, m, ħ)
    end
    resonances    = find_narrow_resonances(p_st, t_st, V, x_L, h, m, ħ)
    p_q, w_q, t_q = resonance_quadrature(p_st, t_st, resonances, V, x_L, h, m, ħ)
    for (p_r, Γ_p) in resonances
        Γ = Γ_p * p_r / m
        @printf("  resonance:    E_r = %.10f, Γ = %.3e, ħ/Γ = %.3e\n", p_r^2 / (2 * m), Γ, ħ / Γ)
    end
    isempty(resonances) || println("  quadrature:   ", length(p_q), " momenta (refined around the resonances)")
    return (p_st = p_st, p_q = p_q, w_q = w_q, t_q = t_q, resonances = resonances, V = V, h = h, n_x = n_x)
end

# Exact delay of one energy from the stationary t(p). Returns (τ_q, O_q, τ_EW, T_q).
function quantum_stationary(p0::Float64, sigma0::Float64, st, τ_grid::AbstractRange{Float64}, par)
    (; ħ, m, x_L, n_σp) = par
    (; p_q, w_q, t_q) = st
    σ_p  = ħ / sigma0
    i_lo = searchsortedfirst(p_q, p0 - n_σp * σ_p)
    i_hi = searchsortedlast(p_q,  p0 + n_σp * σ_p)

    # c = w |ψ̃₀(p)|² t(p), |ψ̃₀(p)|² = σ0/(√π ħ)·exp(−σ0²(p − p0)²/ħ²)
    norm_w = sigma0 / (sqrt(π) * ħ)
    n_w = i_hi - i_lo + 1
    c   = Vector{ComplexF64}(undef, n_w)
    ω   = Vector{Float64}(undef, n_w)
    T_q = 0.0
    @inbounds for j in 1:n_w
        i = i_lo + j - 1
        p = p_q[i]
        w = w_q[i] * norm_w * exp(-sigma0^2 * (p - p0)^2 / ħ^2)
        c[j] = w * t_q[i]
        ω[j] = p^2 / (2 * m * ħ)
        T_q += w * abs2(t_q[i])
    end
    τ_q, O_q = find_max_tau(τ -> quantum_overlap(τ, c, ω, 1 / (2π * ħ)), τ_grid)

    # τ_EW = ħ (m/p0) dφ/dp by a central difference with a step small against the distance
    # to the pole of every narrow resonance
    δp = step(st.p_st)
    for (p_r, Γ_p) in st.resonances
        δp = min(δp, hypot(p0 - p_r, Γ_p / 2) / 20)
    end
    t_plus  = transmission_amplitude(p0 + δp, st.V, x_L, st.h, m, ħ)
    t_minus = transmission_amplitude(p0 - δp, st.V, x_L, st.h, m, ħ)
    τ_EW = ħ * m / p0 * angle(t_plus * conj(t_minus)) / (2 * δp)
    return τ_q, O_q, τ_EW, T_q
end

# Exact delay of one energy by the split-operator propagation to the time T_end.
# ws: per-task buffers ψ, ψ0_p. Returns (τ_q, O_q, τ_EW, T_q).
function quantum_time_dependent!(ws, p0::Float64, sigma0::Float64, x::Vector{Float64}, S::SplitOperator,
                                 n_steps::Int, T_end::Float64, τ_grid::AbstractRange{Float64}, par)
    (; ħ, m, x0_init) = par
    ψ, ψ0_p = ws.ψ, ws.ψ0_p
    p_grid  = S.p
    norm_p  = (x[2] - x[1]) / length(ψ)

    ψ0_p .= gaussian_packet.(x, x0_init, p0, sigma0, ħ)
    S.fft * ψ0_p
    ψ .= gaussian_packet.(x, x0_init, p0, sigma0, ħ)
    for _ in 1:n_steps
        split_step!(ψ, S)
    end
    S.fft * ψ

    T_q = 0.0
    @inbounds for k in eachindex(ψ)
        p_grid[k] > 0 && (T_q += abs2(ψ[k]))
    end
    T_q *= norm_p

    # overlap coefficients with the state propagated freely back by T_end
    amp_cut = 1e-20 * maximum(abs, ψ0_p)
    c = ComplexF64[]
    ω = Float64[]
    @inbounds for k in eachindex(ψ0_p)
        abs(ψ0_p[k]) > amp_cut || continue
        ωk = p_grid[k]^2 / (2 * m * ħ)
        push!(c, norm_p * conj(ψ0_p[k]) * ψ[k] * cis(ωk * T_end))
        push!(ω, ωk)
    end
    τ_q, O_q = find_max_tau(τ -> quantum_overlap(τ, c, ω, 1 / (2π * ħ)), τ_grid)

    # τ_EW from the phase of t(p) = φ̃(p)/ψ̃₀(p), interpolated between the grid nodes around p0
    dp = p_grid[2] - p_grid[1]
    t_amp(k) = ψ[k] * cis(p_grid[k]^2 / (2 * m * ħ) * T_end) / ψ0_p[k]
    dφdp(k)  = angle(t_amp(k + 1) * conj(t_amp(k - 1))) / (2 * dp)
    k0 = floor(Int, p0 / dp) + 1
    f  = (p0 - p_grid[k0]) / dp
    τ_EW = ħ * m / p0 * ((1 - f) * dφdp(k0) + f * dφdp(k0 + 1))
    return τ_q, O_q, τ_EW, T_q
end

# Semiclassical delay of one energy. Returns (τ_c, O_c, T_c, ess_T, tabulated proposal used).
function classical_delay(p0::Float64, sigma0::Float64, n_steps::Int, T_end::Float64,
                         τ_grid::AbstractRange{Float64}, x_free::NTuple{2, Float64}, par)
    (; ħ, m, potential, x0_init, x_R, dt, N_traj) = par
    _, σ_p = wigner_widths(sigma0, ħ)
    prop = importance_proposal(potential, x0_init, p0, σ_p, par.is_fraction, m, ħ, N_traj;
                               x_search_max = x_R - x0_init)
    x, p, w, _, _ = sample_symmetric(MersenneTwister(par.rng_seed), N_traj, x0_init, p0, sigma0, ħ,
                                     prop, par.is_fraction)
    b = advance!(Branches(x, p, w), n_steps, potential, dt, m, ħ; V_threshold = par.V_barrier_threshold,
                 w_min_spawn = par.w_min_spawn, max_gen = par.max_gen, x_free = x_free)

    # transmitted branches: beyond the potential and moving to the right
    transmitted(k) = b.p[k] > 0 && b.x[k] > x_R
    T_c = sum((b.w[k] for k in eachindex(b.w) if transmitted(k)); init = 0.0) / N_traj

    amp  = Float64[]
    ξ    = Float64[]
    v    = Float64[]
    pref = 1 / (π * ħ * N_traj)
    @inbounds for k in eachindex(b.x)
        transmitted(k) || continue
        a_k = pref * b.w[k] * exp(-sigma0^2 * (b.p[k] - p0)^2 / ħ^2)
        a_k > 0 || continue
        push!(amp, a_k)
        push!(ξ, b.x[k] - b.p[k] * T_end / m - x0_init)
        push!(v, b.p[k] / m)
    end
    τ_c, O_c = find_max_tau(τ -> classical_overlap(τ, amp, ξ, v, 1 / sigma0^2), τ_grid)
    return τ_c, O_c, T_c, prop === nothing ? NaN : prop.ess.is_T, prop !== nothing && prop.tab !== nothing
end

function run_width(par, sigma0::Float64)
    (; ħ, m, potential, x0_init, n_energies, E_min, E_max, quantum_method, x_L, x_R, x_final, dt, N_traj) = par
    out_folder = "eisenbud_wigner_$(potential.name)_hbar=$(ħ)_m=$(m)_sigma=$(sigma0)_E=$(E_min)-$(E_max)"

    N_traj % 4 == 0 || error("N_traj must be divisible by 4 (symmetric sampling)")
    0 ≤ par.is_fraction < 1 || error("is_fraction must lie in [0, 1)")
    quantum_method in (:stationary, :time_dependent) ||
        error("quantum_method must be :stationary or :time_dependent")

    x         = collect(LinRange(par.x_min, par.x_max, par.x_num))
    energies  = collect(LinRange(E_min, E_max, n_energies))
    p0_values = [packet_momentum(E, potential, x0_init, m) for E in energies]
    n_steps   = [round(Int, (x_final - x0_init) * m / (p0 * dt)) for p0 in p0_values]
    τ_grid    = par.τ_min:par.τ_step:par.τ_max
    V_top     = maximum(potential, x)
    x_free    = free_region(potential, x_L, x_R)

    println("=== Eisenbud–Wigner time: exact against semiclassical ===")
    println("  potential:    ", potential.name, "  (top V = ", round(V_top; digits = 4), "),  ħ = ", ħ, ",  m = ", m)
    println("  σ0:           ", sigma0)
    println("  energies:     ", n_energies, " values in [", E_min, ", ", E_max, "]")
    println("  quantum:      ", quantum_method)
    println("  end:          x_final = ", x_final, "  (T = ", round(n_steps[end] * dt; digits = 1), " … ",
            round(n_steps[1] * dt; digits = 1), ")")
    println("  trajectories: ", N_traj, "  (max_gen = ", par.max_gen, ", is_fraction = ", par.is_fraction, ")")
    println("  threads:      ", nthreads())
    mkpath(out_folder)

    FFTW.set_num_threads(1)
    if quantum_method == :stationary
        st = stationary_setup(par, sigma0, p0_values, V_top)
    else
        S    = SplitOperator(x, potential.(x) .+ absorbing_potential(x, par.abs_width, par.abs_strength), dt, m, ħ)
        pool = [(ψ = Vector{ComplexF64}(undef, par.x_num), ψ0_p = Vector{ComplexF64}(undef, par.x_num))
                for _ in 1:nthreads()]
        st   = (S = S, pool = pool)
    end

    tau_q   = zeros(n_energies); tau_c   = zeros(n_energies); tau_EW = zeros(n_energies)
    O_q_max = zeros(n_energies); O_c_max = zeros(n_energies)
    T_q     = zeros(n_energies); T_c     = zeros(n_energies)
    ess_T   = zeros(n_energies); tab_used = fill(false, n_energies)

    @time parallel_sweep(n_energies) do ie, slot
        p0    = p0_values[ie]
        T_end = n_steps[ie] * dt
        tau_q[ie], O_q_max[ie], tau_EW[ie], T_q[ie] = quantum_method == :stationary ?
            quantum_stationary(p0, sigma0, st, τ_grid, par) :
            quantum_time_dependent!(st.pool[slot], p0, sigma0, x, st.S, n_steps[ie], T_end, τ_grid, par)
        tau_c[ie], O_c_max[ie], T_c[ie], ess_T[ie], tab_used[ie] =
            classical_delay(p0, sigma0, n_steps[ie], T_end, τ_grid, x_free, par)
    end

    n_nan = count(isnan, tau_q) + count(isnan, tau_c)
    n_nan > 0 && println("  WARNING: ", n_nan, " delays outside [τ_min, τ_max] (NaN)")
    tab_range = any(tab_used) ? (minimum(energies[tab_used]), maximum(energies[tab_used])) : ()
    any(tab_used) && println("  tabulated importance proposal used at ", count(tab_used), " energies, E0 ∈ ", tab_range)

    # ---------- data ----------
    open(joinpath(out_folder, "tau_EW.txt"), "w") do io
        println(io, "# E0\ttau_exact\ttau_semiclassical\ttau_EW_stationary\t",
                    "O_exact\tO_semiclassical\tT_exact\tT_semiclassical\tESS_transmitted_semiclassical")
        for ie in 1:n_energies
            @printf(io, "%.6f\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.6e\n", energies[ie], tau_q[ie],
                    tau_c[ie], tau_EW[ie], O_q_max[ie], O_c_max[ie], T_q[ie], T_c[ie], ess_T[ie])
        end
    end

    entries = Pair{String, Any}["potential" => potential.name, "V_top" => V_top, "hbar" => ħ, "m" => m,
                                "x0" => x0_init, "sigma0" => sigma0, "quantum_method" => quantum_method]
    if quantum_method == :stationary
        push!(entries, "x_stationary" => (x_L, x_R, st.n_x),
                       "p_grid" => (first(st.p_st), last(st.p_st), length(st.p_st)), "n_sigma_p" => par.n_σp)
        for (p_r, Γ_p) in st.resonances
            push!(entries, "resonance_E_Gamma" => (p_r^2 / (2 * m), Γ_p * p_r / m))
        end
    else
        push!(entries, "x_grid" => (par.x_min, par.x_max, par.x_num))
    end
    push!(entries, "x_final" => x_final, "dt" => dt, "N_traj" => N_traj, "is_fraction" => par.is_fraction,
                   "tabulated_proposal_count_Emin_Emax" => (count(tab_used), tab_range...),
                   "x_transmitted" => x_R, "max_gen" => par.max_gen, "rng_seed" => par.rng_seed,
                   "tau_search" => (par.τ_min, par.τ_max, par.τ_step))
    write_parameters(joinpath(out_folder, "parameters.txt"), entries)

    # ---------- figures ----------
    tag    = "$(potential.name), ħ = $(ħ), m = $(m), σ0 = $(sigma0)"
    layout = (size = par.figure_size, dpi = par.figure_dpi, left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
    p_tau = plot(energies, tau_q; label = "exact (Schrödinger)", lw = 2, xlabel = "E0", ylabel = "τ",
                 title = "Eisenbud–Wigner time – " * tag, legend = :topright, layout...)
    plot!(p_tau, energies, tau_c;  label = "semiclassical (TWA + Gamow)", lw = 2)
    plot!(p_tau, energies, tau_EW; label = "stationary ħ dφ/dE", lw = 1.5, ls = :dash)
    hline!(p_tau, [0.0];   color = :gray,  ls = :dot, label = "")
    vline!(p_tau, [V_top]; color = :black, ls = :dot, label = "barrier top")
    savefig(p_tau, joinpath(out_folder, "tau_EW.png"))

    p_T = plot(energies, T_q; label = "exact (Schrödinger)", lw = 2, yscale = :log10, xlabel = "E0",
               ylabel = "T", title = "Transmission probability – " * tag, legend = :bottomright, layout...)
    plot!(p_T, energies, T_c; label = "semiclassical (TWA + Gamow)", lw = 2)
    vline!(p_T, [V_top]; color = :black, ls = :dot, label = "barrier top")
    savefig(p_T, joinpath(out_folder, "transmission.png"))

    println("Done. Output in: ", out_folder, "\n")
end

main(par) = foreach(sigma0 -> run_width(par, Float64(sigma0)), par.sigmas)

if abspath(PROGRAM_FILE) == @__FILE__
    main(params)
end
