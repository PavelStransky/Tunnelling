# Overlap and fidelity of the exact Wigner function (stationary scattering states, Numerov) with the
# semiclassical one (TWA with Gamow tunnelling) as functions of time and of the packet energy;
# the Wigner function is evaluated in the points of the trajectories. See README.md.
# Run:  julia --project -t auto exact_twa_overlap.jl

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
    sigmas  = (0.5, 1.0, 3.0),

    # energy sweep
    n_energies = 301,
    E_min      = 0.2,
    E_max      = 1.7,

    # time axis: :scaled – t ∈ [0, T(E0)], T = (x_final − x0)m/p0 (regular axis of x_free = x0 + p0t/m);
    # :absolute – common t ∈ [0, t_max] (t_max = 0: T(E_min))
    time_axis = :scaled,
    x_final   = 60.0,
    t_max     = 0.0,
    n_times   = 201,
    x_free    = nothing,        # a number: only t = 0 and the time at which the free packet is at
                                # x_free are computed (overrides time_axis, x_final, n_times)

    # stationary states; V must be negligible outside [x_L, x_R]
    x_L = -15.0,
    x_R =  15.0,
    numerov_per_λ = 100,        # Numerov step h ≤ λ_min/numerov_per_λ
    dp_per_σp     = 20,         # momentum step ħ/(σ0·dp_per_σp), refined if needed
    n_σp          = 12.0,       # momentum window p0 ± n_σp·ħ/σ0
    κ_res         = 0.02,       # node spacing ≈ κ_res·|p − p_r| near a narrow resonance

    # grid of the Wigner function
    p_margin   = 1.05,          # momentum range ±p_margin·(largest momentum of ψ)
    support_nσ = 6.0,           # support of the packet ±support_nσ·σ(t)
    n_pw       = 8,             # momentum step ≤ (ħ/√2σ0)/n_pw
    n_int      = 4,             # interpolation of W in the trajectory points: 4 cubic, 2 linear

    # "left"/"right" parts: smooth window of half-width ℓ_split around x_split
    x_split = 30.0,
    ℓ_split = 0.5,

    # trajectories
    dt                  = 0.1,          # largest time step (times scale as √m)
    N_traj              = 1_000_000,    # divisible by 4
    rng_seed            = 42,           # the same for all energies
    is_fraction         = 0.5,          # fraction of momenta drawn from the importance proposal (0 = off)
    max_gen             = 50,
    w_min_spawn         = 0.0,
    V_barrier_threshold = 0.05,
    neff_min            = 10.0,         # FF = NaN where fewer effective classical points contribute

    # figures
    figure_size = (1400, 900),
    figure_dpi  = 150,
)

# ============================================================================
# Code
# ============================================================================

# ----------------------------------------------------------------------------
# Exact state from stationary scattering states
# ----------------------------------------------------------------------------

# Momentum nodes and weights: the uniform grid p_st, condensed around narrow resonances by the
# smooth map dp/ds = [dp⁻² + Σ_r κ⁻²((p − p_r)² + (Γ_r/2)²)⁻¹]^(−1/2), s = 0, 1, 2, …
function mapped_nodes(p_st::AbstractRange{Float64}, resonances::Vector{NTuple{2, Float64}};
                      κ::Float64 = 0.02, n_rk::Int = 4)
    dp = step(p_st)
    isempty(resonances) && return collect(p_st), fill(dp, length(p_st))

    h(p) = 1 / sqrt(1 / dp^2 + sum(r -> 1 / (κ^2 * ((p - r[1])^2 + (r[2] / 2)^2)), resonances))
    p_q = Float64[]
    w_q = Float64[]
    p   = first(p_st)
    ds  = 1.0 / n_rk
    while p ≤ last(p_st)
        push!(p_q, p)
        push!(w_q, h(p))
        for _ in 1:n_rk
            k1 = h(p)
            k2 = h(p + 0.5 * ds * k1)
            k3 = h(p + 0.5 * ds * k2)
            k4 = h(p + ds * k3)
            p += ds * (k1 + 2k2 + 2k3 + k4) / 6
        end
    end
    return p_q, w_q
end

# P[i, j] = exp(i k_j x_i), x_i = x_first + (i − 1)dx
function plane_waves!(P::AbstractMatrix{ComplexF64}, x_first::Float64, dx::Float64,
                      k::AbstractVector{Float64})
    nr = size(P, 1)
    @inbounds for j in eachindex(k)
        z = cis(k[j] * dx)
        e = cis(k[j] * x_first)
        for i in 1:nr
            P[i, j] = e
            e *= z
        end
    end
    return P
end

# Ψ[i, n] = ψ(x_s + (i − 1)dx, t_axis[n]) = Σ_j c_j(t_n) ψ_{p_j}(x_i): plane waves left of node iL and
# right of node iR, the Numerov states sc.χ in between, free plane waves for the nodes p_neg ≤ 0.
function quantum_snapshots(p0::Float64, sigma0::Float64, t_axis::Vector{Float64}, sc,
                           i_lo::Int, i_hi::Int, p_neg::Vector{Float64},
                           x_s::Float64, N_x::Int, iL::Int, iR::Int, par; nb::Int = 128)
    (; ħ, m) = par
    x0   = par.x0_init
    dx   = sc.dx
    N_t  = length(t_axis)
    nw   = i_hi - i_lo + 1
    amp0 = (sigma0^2 / (π * ħ^2))^0.25 / sqrt(2π * ħ)

    # c_j(t_n) = w_j ψ̃₀(p_j) e^{−iE_j t_n/ħ}/√(2πħ)
    function coefficients(ps, ws)
        C = Matrix{ComplexF64}(undef, length(ps), N_t)
        @inbounds for j in eachindex(ps)
            p = ps[j]
            E = p^2 / (2 * m)
            a = ws[j] * amp0 * exp(-sigma0^2 * (p - p0)^2 / (2 * ħ^2)) * cis(-p * x0 / ħ)
            for n in 1:N_t
                C[j, n] = a * cis(-E * t_axis[n] / ħ)
            end
        end
        return C
    end

    pw  = view(sc.p_q, i_lo:i_hi)
    k   = pw ./ ħ
    C   = coefficients(pw, view(sc.w_q, i_lo:i_hi))
    D   = similar(C)
    Ψ   = zeros(ComplexF64, N_x, N_t)
    P   = Matrix{ComplexF64}(undef, nb, nw)
    Tmp = Matrix{ComplexF64}(undef, nb, N_t)

    # left: Σ c (e^{ikx} + r e^{−ikx}) = P·C + conj(P·conj(r·C))
    D .= conj.(view(sc.r_q, i_lo:i_hi) .* C)
    i1 = 1
    while i1 ≤ iL - 1
        i2 = min(i1 + nb - 1, iL - 1); nr = i2 - i1 + 1
        Pv = view(P, 1:nr, :)
        plane_waves!(Pv, x_s + (i1 - 1) * dx, dx, k)
        Ψv = view(Ψ, i1:i2, :)
        mul!(Ψv, Pv, C)
        Tv = view(Tmp, 1:nr, :)
        mul!(Tv, Pv, D)
        Ψv .+= conj.(Tv)
        i1 = i2 + 1
    end

    # inside: Numerov states
    mul!(view(Ψ, iL:iR, :), view(sc.χ_q, :, i_lo:i_hi), C)

    # right: Σ c t e^{ikx}
    D .= view(sc.t_q, i_lo:i_hi) .* C
    i1 = iR + 1
    while i1 ≤ N_x
        i2 = min(i1 + nb - 1, N_x); nr = i2 - i1 + 1
        Pv = view(P, 1:nr, :)
        plane_waves!(Pv, x_s + (i1 - 1) * dx, dx, k)
        mul!(view(Ψ, i1:i2, :), Pv, D)
        i1 = i2 + 1
    end

    # components p ≤ 0: free plane waves everywhere
    if !isempty(p_neg)
        Cn = coefficients(p_neg, fill(sc.dp_st, length(p_neg)))
        kn = p_neg ./ ħ
        Pn = Matrix{ComplexF64}(undef, nb, length(p_neg))
        i1 = 1
        while i1 ≤ N_x
            i2 = min(i1 + nb - 1, N_x); nr = i2 - i1 + 1
            Pv = view(Pn, 1:nr, :)
            plane_waves!(Pv, x_s + (i1 - 1) * dx, dx, kn)
            mul!(view(Ψ, i1:i2, :), Pv, Cn, true, true)
            i1 = i2 + 1
        end
    end
    return Ψ
end

# ----------------------------------------------------------------------------
# Wigner function in the trajectory points
# ----------------------------------------------------------------------------

# Row W(x_r, p_l), p_l = (l − 1 − M)·dp_w, l = 1 … 2M, into column `slot` of Wring (needs ψ[r−M … r+M])
@inline function wigner_row!(Wring::Matrix{Float64}, slot::Int, ψ::AbstractVector{ComplexF64},
                             r::Int, M::Int, buf::Vector{ComplexF64}, plan, scale::Float64)
    N_y = 2M
    @inbounds for j in 0:M-1
        buf[j + 1] = conj(ψ[r - j]) * ψ[r + j]
    end
    @inbounds for j in 1:M
        buf[N_y - j + 1] = conj(ψ[r + j]) * ψ[r - j]
    end
    plan * buf
    @inbounds for idx in 1:M
        Wring[idx, slot] = scale * real(buf[idx + M])
    end
    @inbounds for idx in M+1:N_y
        Wring[idx, slot] = scale * real(buf[idx - M])
    end
    return nothing
end

# Slot of row q in the ring buffer (computed if it is not there)
@inline function ensure_row!(q::Int, Wring, slot_row, ψ, M, buf, plan, scale)
    slot = mod(q, length(slot_row)) + 1
    if slot_row[slot] != q
        wigner_row!(Wring, slot, ψ, q, M, buf, plan, scale)
        slot_row[slot] = q
    end
    return slot
end

# C^∞ step from 0 (x ≤ x_split − ℓ) to 1 (x ≥ x_split + ℓ)
@inline function split_window(x::Float64, x_split::Float64, ℓ::Float64)
    u = (x - x_split + ℓ) / (2ℓ)
    u ≤ 0.0 && return 0.0
    u ≥ 1.0 && return 1.0
    a = exp(-1 / u)
    b = exp(-1 / (1 - u))
    return a / (a + b)
end

# Lagrange weights at f ∈ [0, 1): n_int = 2 (nodes 0, 1) or 4 (nodes −1, 0, 1, 2)
@inline function lagrange_weights(n_int::Int, f::Float64)
    if n_int == 2
        return (1 - f, f, 0.0, 0.0)
    else
        return (-f * (f - 1) * (f - 2) / 6, (f + 1) * (f - 1) * (f - 2) / 2,
                -(f + 1) * f * (f - 2) / 2, (f + 1) * f * (f - 1) / 6)
    end
end

# One time slice: Σ_k v_k W(x_k, p_k), Σ_k v_k, Σ_k a_k v_k², Σ_k v_k² over the branches b with the
# weights v_k = w_k (side = 0), w_k(1 − f(x_k)) (side = −1) or w_k f(x_k) (side = +1), f = split_window;
# W is the Wigner function of ψ (grid x_s + (i − 1)dx), interpolated in the sheared coordinates
# (x − pt/m, p). Rows of W are computed in ascending order and kept in the ring buffer ws.Wring.
function snapshot_overlap!(ψ::AbstractVector{ComplexF64}, x_s::Float64, t::Float64, side::Int,
                           b::Branches, a0::Vector{Float64}, ws, sc, par)
    (; m, x_split, ℓ_split, n_int) = par
    (; dx, M, dp_w) = sc
    (; key, perm, cnt, Wring, slot_row, buf, plan) = ws
    N_x     = length(ψ)
    N_y     = 2M
    scale   = dx / (π * par.ħ)
    r_lo    = M + 1
    r_hi    = N_x - M
    shear   = t / m
    Δ       = dp_w * shear / dx
    p_first = -M * dp_w
    hh      = n_int ÷ 2
    floor(Int, (n_int - 1) * Δ) + n_int + 1 ≤ length(slot_row) || error("Wring has too few rows")

    # norms and sort keys (lowest row needed by a branch)
    Nc = 0.0
    Sc = 0.0
    Q2 = 0.0
    fill!(cnt, 0)
    @inbounds for k in eachindex(b.x)
        key[k] = 0
        x = b.x[k]
        w = b.w[k]
        if side != 0
            f = split_window(x, x_split, ℓ_split)
            w *= side > 0 ? f : 1 - f
        end
        p = b.p[k]
        Nc += w
        Sc += a0[b.root[k]] * w * w
        Q2 += w * w
        w == 0.0 && continue
        u  = (p - p_first) / dp_w
        u ≥ 0.0 || continue
        l0 = floor(Int, u) + 1
        (l0 - hh + 1 ≥ 1 && l0 + hh ≤ N_y) || continue
        v_lo  = (x - (p - (p_first + (l0 - hh) * dp_w)) * shear - x_s) / dx
        r_min = floor(Int, v_lo) + 1 - (hh - 1)
        r_max = floor(Int, v_lo + (n_int - 1) * Δ) + 1 + hh
        (r_min ≥ r_lo && r_max ≤ r_hi) || continue
        key[k] = r_min
        cnt[r_min] += 1
    end

    # counting sort by row
    total = 0
    @inbounds for r in 1:N_x
        c = cnt[r]; cnt[r] = total; total += c
    end
    @inbounds for k in eachindex(b.x)
        r = key[k]
        r == 0 && continue
        cnt[r] += 1
        perm[cnt[r]] = k
    end

    # interpolation
    fill!(slot_row, 0)
    O = 0.0
    @inbounds for idx in 1:total
        k = perm[idx]
        x = b.x[k]; p = b.p[k]; w = b.w[k]
        if side != 0
            f = split_window(x, x_split, ℓ_split)
            w *= side > 0 ? f : 1 - f
        end
        u    = (p - p_first) / dp_w
        l0   = floor(Int, u) + 1
        wp   = lagrange_weights(n_int, u - (l0 - 1))
        v_lo = (x - (p - (p_first + (l0 - hh) * dp_w)) * shear - x_s) / dx

        Wk = 0.0
        for a in 1:n_int
            c  = l0 - hh + a
            v  = v_lo + (a - 1) * Δ
            rc = floor(Int, v) + 1
            wx = lagrange_weights(n_int, v - (rc - 1))
            s  = 0.0
            for bb in 1:n_int
                slot = ensure_row!(rc - hh + bb, Wring, slot_row, ψ, M, buf, plan, scale)
                s += wx[bb] * Wring[c, slot]
            end
            Wk += wp[a] * s
        end
        O += w * Wk
    end
    return O, Nc, Sc, Q2
end

# ----------------------------------------------------------------------------
# One energy
# ----------------------------------------------------------------------------

# Returns (t_axis, res, number of branches, deepest generation, ess_T, tabulated proposal used);
# res[n, :] = (O_total, O_left, O_right, Nq_left, Nq_right, Nc_left, Nc_right,
#              Wc2_total, Wc2_left, Wc2_right, Neff_total, Neff_left, Neff_right) at t_axis[n]
function energy_series!(ws, p0::Float64, T_end::Float64, x_sup::Float64, sigma0::Float64, sc, par)
    (; ħ, m, potential, x0_init, x_split, ℓ_split, n_σp, N_traj) = par
    (; dx, M, n_times) = sc

    # n_times slices with sps Verlet steps (dt ≤ par.dt) between them
    sps    = max(1, ceil(Int, T_end / ((n_times - 1) * par.dt)))
    dt     = T_end / ((n_times - 1) * sps)
    t_axis = [(n - 1) * sps * dt for n in 1:n_times]

    # ψ grid: rows of W cover [−x_sup, x_sup], plus the margin M on both sides
    i_s = floor(Int, -x_sup / dx) - M - 1
    i_e = ceil(Int,   x_sup / dx) + M + 2
    N_x = i_e - i_s + 1
    x_s = i_s * dx
    iL  = sc.iL_g - i_s + 1
    iR  = sc.iR_g - i_s + 1
    (1 < iL ≤ iR < N_x) || error("the interval [x_L, x_R] lies outside the ψ grid")

    σ_p   = ħ / sigma0
    i_lo  = searchsortedfirst(sc.p_q, p0 - n_σp * σ_p)
    i_hi  = searchsortedlast(sc.p_q,  p0 + n_σp * σ_p)
    p_neg = (sc.neg_allowed && p0 - n_σp * σ_p ≤ 0) ?
            [-j * sc.dp_st for j in 0:floor(Int, (n_σp * σ_p - p0) / sc.dp_st)] : Float64[]
    Ψ = quantum_snapshots(p0, sigma0, t_axis, sc, i_lo, i_hi, p_neg, x_s, N_x, iL, iR, par)

    # classical branches and their Liouville factors a = W₀(z₀)/w₀ = c·ρ_x(x)·q(p)
    σ_xc, σ_pc = wigner_widths(sigma0, ħ)
    prop = importance_proposal(potential, x0_init, p0, σ_pc, par.is_fraction, m, ħ, N_traj;
                               x_search_max = par.x_R - x0_init)
    x, p, w, c, α = sample_symmetric(MersenneTwister(par.rng_seed), N_traj, x0_init, p0, sigma0, ħ,
                                     prop, par.is_fraction)
    a0 = [c * exp(-(x[k] - x0_init)^2 / (2 * σ_xc^2)) / (sqrt(2π) * σ_xc) *
          ((1 - α) * exp(-(p[k] - p0)^2 / (2 * σ_pc^2)) / (sqrt(2π) * σ_pc) +
           α * (prop === nothing ? 0.0 : proposal_density(prop, p[k]))) for k in eachindex(x)]
    b = Branches(x, p, w)

    resize!(ws.cnt, N_x)
    fwin = [split_window(x_s + (i - 1) * dx, x_split, ℓ_split) for i in 1:N_x]
    aL   = sqrt.(1 .- fwin)
    aR   = sqrt.(fwin)
    ψL   = zeros(ComplexF64, N_x)
    ψR   = zeros(ComplexF64, N_x)
    res  = zeros(Float64, n_times, 13)

    for n in 1:n_times
        n > 1 && advance!(b, (n - 1) * sps, potential, dt, m, ħ; V_threshold = par.V_barrier_threshold,
                          w_min_spawn = par.w_min_spawn, max_gen = par.max_gen, x_free = sc.x_free)
        ψ = view(Ψ, :, n)
        ψL .= aL .* ψ
        ψR .= aR .* ψ
        resize!(ws.key, length(b)); resize!(ws.perm, length(b))
        O_t, NcT, S_t, Q_t = snapshot_overlap!(ψ,  x_s, t_axis[n],  0, b, a0, ws, sc, par)
        O_l, NcL, S_l, Q_l = snapshot_overlap!(ψL, x_s, t_axis[n], -1, b, a0, ws, sc, par)
        O_r, NcR, S_r, Q_r = snapshot_overlap!(ψR, x_s, t_axis[n], +1, b, a0, ws, sc, par)

        res[n, 1]  = O_t / N_traj
        res[n, 2]  = O_l / N_traj
        res[n, 3]  = O_r / N_traj
        res[n, 4]  = sum(abs2, ψL) * dx
        res[n, 5]  = sum(abs2, ψR) * dx
        res[n, 6]  = NcL / N_traj
        res[n, 7]  = NcR / N_traj
        res[n, 8]  = S_t / N_traj
        res[n, 9]  = S_l / N_traj
        res[n, 10] = S_r / N_traj
        res[n, 11] = Q_t > 0 ? NcT^2 / Q_t : 0.0
        res[n, 12] = Q_l > 0 ? NcL^2 / Q_l : 0.0
        res[n, 13] = Q_r > 0 ? NcR^2 / Q_r : 0.0
    end
    return t_axis, res, length(b), maximum(b.gen), prop === nothing ? NaN : prop.ess.is_T,
           prop !== nothing && prop.tab !== nothing
end

# ----------------------------------------------------------------------------
function run_width(par, sigma0::Float64)
    (; ħ, m, potential, x0_init, n_energies, E_min, E_max, x_L, x_R, n_σp, n_int, x_split, ℓ_split,
       N_traj, neff_min) = par
    x_free_mode = par.x_free !== nothing
    time_axis   = x_free_mode ? :scaled : par.time_axis
    x_final     = x_free_mode ? Float64(par.x_free) : par.x_final
    n_times     = x_free_mode ? 2 : par.n_times
    out_folder  = "exact_twa_overlap_$(potential.name)_hbar=$(ħ)_m=$(m)_sigma=$(sigma0)_E=$(E_min)-$(E_max)" *
                  (time_axis == :absolute ? "_abs" : "") * (x_free_mode ? "_xfree=$(x_final)" : "")

    N_traj % 4 == 0 || error("N_traj must be divisible by 4 (symmetric sampling)")
    0 ≤ par.is_fraction < 1 || error("is_fraction must lie in [0, 1)")
    time_axis in (:scaled, :absolute) || error("time_axis must be :scaled or :absolute")
    x_final > x0_init || error("x_final = $x_final must lie right of x0 = $x0_init")
    BLAS.set_num_threads(1)
    FFTW.set_num_threads(1)

    energies  = collect(LinRange(E_min, E_max, n_energies))
    p0_values = [packet_momentum(E, potential, x0_init, m) for E in energies]
    T_free(p0) = (x_final - x0_init) * m / p0
    T_ends = time_axis == :scaled ? T_free.(p0_values) :
             fill(par.t_max > 0 ? par.t_max : T_free(p0_values[1]), n_energies)
    x_ends = [x0_init + p0 * T / m for (p0, T) in zip(p0_values, T_ends)]
    σ_T    = [free_width(sigma0, T, m, ħ) for T in T_ends]
    # at T the transmitted packet is near x_end, the reflected one near −x_end
    x_sups = [max(xe, abs(x0_init)) + par.support_nσ * s for (xe, s) in zip(x_ends, σ_T)]

    # ψ grid = y grid of the Wigner function
    σ_p  = ħ / sigma0
    p_hi = maximum(p0_values) + n_σp * σ_p
    dx   = π * ħ / (2 * par.p_margin * p_hi)
    y_req  = max(par.support_nσ * maximum(σ_T), π * ħ / (2 * (ħ / (sqrt(2) * sigma0) / par.n_pw)))
    N_y    = nextpow(2, ceil(Int, 2 * y_req / dx))
    M      = N_y ÷ 2
    y_max  = M * dx
    dp_w   = π * ħ / (N_y * dx)
    R_ring = floor(Int, (n_int - 1) * dp_w * maximum(T_ends) / (m * dx)) + n_int + 2

    # uniform momentum grid whose period 2πħ/dp in x exceeds the interval where ψ is needed
    L_span   = 2 * (maximum(x_sups) + y_max)
    dp_alias = 2π * ħ / (1.2 * L_span)
    dp_st    = min(σ_p / par.dp_per_σp, dp_alias)
    p_lo     = max(minimum(p0_values) - n_σp * σ_p, dp_st)
    p_st     = p_lo:dp_st:p_hi
    neg_allowed = p_lo == dp_st     # grid aligned with multiples of dp: nodes p ≤ 0 can be added

    # Numerov grid aligned with the ψ grid (x = i·dx)
    iL_g  = floor(Int, x_L / dx)
    iR_g  = ceil(Int,  x_R / dx)
    xL    = iL_g * dx
    xR    = iR_g * dx
    n_in  = iR_g - iL_g + 1
    n_sub = ceil(Int, dx / (2π * ħ / p_hi / par.numerov_per_λ))
    h     = dx / n_sub
    V     = [potential(xL + (j - 1) * h) for j in 1:(n_in - 1) * n_sub + 1]
    V_top = maximum(potential, range(-60.0, 60.0; length = 200_001))
    x_free = free_region(potential, x_L, x_R)

    println("=== Overlap of the exact Wigner function with the TWA ===")
    println("  potential:    ", potential.name, "  (top V = ", round(V_top; digits = 4), "),  ħ = ", ħ, ",  m = ", m)
    println("  σ0:           ", sigma0)
    println("  energies:     ", n_energies, " values in [", E_min, ", ", E_max, "]")
    println("  time:         ", time_axis, ", ", n_times, " slices, T = ", round(minimum(T_ends); digits = 1),
            " … ", round(maximum(T_ends); digits = 1))
    @printf("  ψ grid:       dx = %.4f, x ∈ ±%.1f (%d … %d points)\n", dx, maximum(x_sups) + y_max,
            round(Int, 2 * (minimum(x_sups) + y_max) / dx), round(Int, L_span / dx))
    @printf("  Wigner:       N_y = %d, y_max = %.1f, dp_w = %.4f (p ∈ ±%.2f)\n", N_y, y_max, dp_w, M * dp_w)
    println("  Numerov:      ", length(V), " points on [", round(xL; digits = 3), ", ", round(xR; digits = 3),
            "], h = dx/", n_sub)
    dp_st == dp_alias && println("  momentum step refined to dp = ", dp_st)
    neg_allowed && println("  the momentum window reaches p ≤ 0 – these components move freely")
    println("  trajectories: ", N_traj, "  (max_gen = ", par.max_gen, ", is_fraction = ", par.is_fraction,
            "), free flight outside [", round(x_free[1]; digits = 2), ", ", round(x_free[2]; digits = 2), "]")
    println("  split:        x_split = ", round(x_split; digits = 3), " ± ", ℓ_split)
    x_split - ℓ_split < x_free[2] &&
        println("  WARNING: the window [x_split − ℓ, x_split + ℓ] reaches into the potential")
    println("  threads:      ", nthreads())
    V_edge = max(abs(V[1]), abs(V[end]))
    V_top > 0 && V_edge > 1e-12 * V_top &&
        println("  WARNING: V at the edge of [x_L, x_R] is ", V_edge, " – widen x_L, x_R")
    tail0 = exp(-(sigma0 * minimum(p0_values) / ħ)^2 / 2)
    tail0 > 1e-6 &&
        @printf("  WARNING: |ψ̃₀(0)|/max = %.1e – the sharp cut at p = 0 gives tails of this order in x\n", tail0)
    mkpath(out_folder)

    # ---------- scattering data in all momentum nodes ----------
    t_st = Vector{ComplexF64}(undef, length(p_st))
    @threads for i in eachindex(p_st)
        t_st[i] = transmission_amplitude(p_st[i], V, xL, h, m, ħ)
    end
    resonances = find_narrow_resonances(p_st, t_st, V, xL, h, m, ħ)
    p_q, w_q   = mapped_nodes(p_st, resonances; κ = par.κ_res)
    for (p_r, Γ_p) in resonances
        Γ = Γ_p * p_r / m
        @printf("  resonance:    E_r = %.10f, Γ = %.3e, ħ/Γ = %.3e\n", p_r^2 / (2 * m), Γ, ħ / Γ)
    end
    N_q = length(p_q)
    t_q = Vector{ComplexF64}(undef, N_q)
    r_q = Vector{ComplexF64}(undef, N_q)
    χ_q = Matrix{ComplexF64}(undef, n_in, N_q)
    @time @threads for i in 1:N_q
        t_q[i], r_q[i] = numerov_scattering!(view(χ_q, :, i), p_q[i], V, xL, h, m, ħ; n_sub = n_sub)
    end
    println("  momenta:      ", N_q, " nodes (", length(p_st), " uniform, dp = ", round(dp_st; sigdigits = 4), ")")

    sc = (p_q = p_q, w_q = w_q, t_q = t_q, r_q = r_q, χ_q = χ_q, dp_st = dp_st, neg_allowed = neg_allowed,
          iL_g = iL_g, iR_g = iR_g, dx = dx, M = M, dp_w = dp_w, x_free = x_free, n_times = n_times)

    # ---------- energy sweep ----------
    pool = map(1:nthreads()) do _
        buf = Vector{ComplexF64}(undef, N_y)
        (buf = buf, plan = plan_fft!(buf; flags = FFTW.MEASURE), Wring = Matrix{Float64}(undef, N_y, R_ring),
         slot_row = zeros(Int, R_ring), key = Int[], perm = Int[], cnt = Int[])
    end
    t_mat      = zeros(n_energies, n_times)
    series     = [zeros(n_energies, n_times) for _ in 1:13]
    n_branches = zeros(Int, n_energies)
    gen_max    = zeros(Int, n_energies)
    ess_T      = zeros(n_energies)
    tab_used   = fill(false, n_energies)

    @time parallel_sweep(n_energies) do ie, slot
        t_axis, res, n_branches[ie], gen_max[ie], ess_T[ie], tab_used[ie] =
            energy_series!(pool[slot], p0_values[ie], T_ends[ie], x_sups[ie], sigma0, sc, par)
        t_mat[ie, :] = t_axis
        for (k, S) in enumerate(series)
            S[ie, :] = @view res[:, k]
        end
    end
    O_tot, O_L, O_R, Nq_L, Nq_R, Nc_L, Nc_R, Wc2_T, Wc2_L, Wc2_R, Neff_T, Neff_L, Neff_R = series

    # ---------- derived quantities ----------
    # FF = O/(‖W_q‖‖W_c‖), ‖W_q‖ = N_q/√(2πħ) (pure state), ‖W_c‖ = √(∫∫ W_c²)
    Nq_tot = Nq_L .+ Nq_R
    fidelity(O, Nq, Wc2, Neff) =
        @. ifelse(Nq > 0 && Wc2 > 0 && Neff ≥ neff_min, O * sqrt(2π * ħ) / (Nq * sqrt(Wc2)), NaN)
    FF_tot = fidelity(O_tot, Nq_tot, Wc2_T, Neff_T)
    FF_L   = fidelity(O_L, Nq_L, Wc2_L, Neff_L)
    FF_R   = fidelity(O_R, Nq_R, Wc2_R, Neff_R)
    x_free_axis = [x0_init + p0_values[ie] * t_mat[ie, n] / m for ie in 1:n_energies, n in 1:n_times]

    @printf("  check: |N_q − 1| ≤ %.1e,  |FF(t=0) − 1| ≤ %.1e,  max FF = %.4f,  at most %d branches\n",
            maximum(abs.(Nq_tot .- 1)), maximum(abs.(FF_tot[:, 1] .- 1)),
            maximum(v -> isnan(v) ? -Inf : v, [FF_tot FF_L FF_R]), maximum(n_branches))
    ess_ok = filter(!isnan, ess_T)
    isempty(ess_ok) || @printf("  effective number of trajectories in the transmitted part: %.3g … %.3g (of %d)\n",
                               minimum(ess_ok), maximum(ess_ok), N_traj)
    tab_range = any(tab_used) ? (minimum(energies[tab_used]), maximum(energies[tab_used])) : ()
    any(tab_used) && println("  tabulated importance proposal used at ", count(tab_used), " energies, E0 ∈ ", tab_range)
    maximum(gen_max) > 1 &&
        println("  WARNING: branches of generation ", maximum(gen_max), " (well between barriers) – ",
                "∫∫ W_c² and FF are only approximate, the images of the branches may overlap")

    # ---------- data ----------
    open(joinpath(out_folder, "xyz.txt"), "w") do io
        println(io, "# Regular grid: ", time_axis == :scaled ? "(x_free, E)" : "(t, E)",
                    "; x_free = x0 + p0 t/m is the position of the free packet")
        println(io, "# FF = O/(NN_q*NN_c), NN_q^2 = int W_q^2 = Nq^2/(2*pi*hbar), NN_c^2 = int W_c^2 = Wc2;",
                    " FF = NaN where Neff < ", neff_min)
        println(io, "# x_free\tt\tE\tFF_total\tFF_left\tFF_right\tO_total\tO_left\tO_right\t",
                    "Nq_left\tNq_right\tNc_left\tNc_right\tWc2_total\tWc2_left\tWc2_right\t",
                    "Neff_total\tNeff_left\tNeff_right")
        for ie in 1:n_energies, n in 1:n_times
            @printf(io, "%.6f\t%.6f\t%.6f", x_free_axis[ie, n], t_mat[ie, n], energies[ie])
            for A in (FF_tot, FF_L, FF_R, O_tot, O_L, O_R, Nq_L, Nq_R, Nc_L, Nc_R,
                      Wc2_T, Wc2_L, Wc2_R, Neff_T, Neff_L, Neff_R)
                @printf(io, "\t%.8e", A[ie, n])
            end
            println(io)
        end
    end

    open(joinpath(out_folder, "final.txt"), "w") do io
        println(io, "# Values at the final time t = T(E0)")
        println(io, "# ess_T = effective number of trajectories in the transmitted part (importance sampling)")
        println(io, "# E0\tT_end\tFF_total\tFF_left\tFF_right\tNq_right\tNc_right\tO_right\tWc2_right\tess_T")
        for ie in 1:n_energies
            @printf(io, "%.6f\t%.6f\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.6e\n",
                    energies[ie], t_mat[ie, end], FF_tot[ie, end], FF_L[ie, end], FF_R[ie, end],
                    Nq_R[ie, end], Nc_R[ie, end], O_R[ie, end], Wc2_R[ie, end], ess_T[ie])
        end
    end

    x_free_mode &&
        write_columns(joinpath(out_folder, @sprintf("FF_xfree=%g.txt", x_final)),
                      ["E0", "t", "FF_total", "FF_left", "FF_right"],
                      [energies, t_mat[:, end], FF_tot[:, end], FF_L[:, end], FF_R[:, end]])

    entries = Pair{String, Any}[
        "potential" => potential.name, "V_top" => V_top, "hbar" => ħ, "m" => m, "x0" => x0_init,
        "sigma0" => sigma0, "time_axis" => time_axis, "x_final" => x_final,
        "T_end_range" => (minimum(T_ends), maximum(T_ends)), "n_times" => n_times,
        "x_stationary" => (xL, xR, length(V)), "p_grid" => (first(p_st), last(p_st), length(p_st)),
        "p_nodes" => N_q, "n_sigma_p" => n_σp]
    for (p_r, Γ_p) in resonances
        push!(entries, "resonance_E_Gamma" => (p_r^2 / (2 * m), Γ_p * p_r / m))
    end
    push!(entries, "dx" => dx, "N_y" => N_y, "y_max" => y_max, "dp_w" => dp_w, "n_int" => n_int,
                   "x_split" => (x_split, ℓ_split), "x_free_classical" => x_free, "dt_max" => par.dt,
                   "N_traj" => N_traj, "is_fraction" => par.is_fraction,
                   "tabulated_proposal_count_Emin_Emax" => (count(tab_used), tab_range...),
                   "max_gen" => (par.max_gen, "(reached $(maximum(gen_max)))"), "rng_seed" => par.rng_seed,
                   "neff_min" => neff_min)
    write_parameters(joinpath(out_folder, "parameters.txt"), entries)

    # ---------- figures ----------
    tag    = "$(potential.name), σ0 = $(sigma0)"
    layout = (size = par.figure_size, dpi = par.figure_dpi, left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
    if x_free_mode
        p = plot(energies, [FF_tot[:, end] FF_L[:, end] FF_R[:, end]]; label = ["FF_total" "FF_left" "FF_right"],
                 xlabel = "E0", ylabel = "FF", title = "FF at x_free = $(x_final) – $(tag)", lw = 1.5,
                 framestyle = :box, layout...)
        savefig(p, joinpath(out_folder, "FF_xfree.png"))
    else
        X_axis  = time_axis == :scaled ? x_free_axis[1, :] : t_mat[1, :]
        X_label = time_axis == :scaled ? "x_free = x0 + p0 t/m" : "t"
        for (file, Z, title, kw) in
                (("FF_total.png", FF_tot, "FF_total = ∫∫W_qW_c / (‖W_q‖‖W_c‖)", (clims = (0, 1),)),
                 ("FF_left.png", FF_L, "FF_left (x ≲ $(x_split))", (clims = (0, 1),)),
                 ("FF_right.png", FF_R, "FF_right (x ≳ $(x_split))", (clims = (0, 1),)),
                 ("Nq_right.png", log10.(max.(Nq_R, 1e-300)), "log10 N_q,right", (;)),
                 ("Nc_over_Nq_right.png", log10.(max.(Nc_R, 1e-300) ./ max.(Nq_R, 1e-300)),
                  "log10 (N_c,right / N_q,right)", (;)))
            save_heatmap(joinpath(out_folder, file), X_axis, energies, Z; xlabel = X_label, ylabel = "E0",
                         title = "$(title) – $(tag)", layout..., kw...)
        end
    end

    println("Done. Output in: ", out_folder, "\n")
end

main(par) = foreach(sigma0 -> run_width(par, Float64(sigma0)), par.sigmas)

if abspath(PROGRAM_FILE) == @__FILE__
    main(params)
end
