# Data for the figures of the paper (plotted in Origin): exact Wigner function Wq(x, p, t),
# semiclassical Wc(x, p, t) (TWA with Gamow tunnelling) on the same grid, their marginals, energy
# contours and the potential, at a few selected times. See README.md.
# Run:  julia --project -t auto wigner_origin.jl

include("common.jl")

# ============================================================================
# Parameters
# ============================================================================
params = (
    ħ = 1.0,
    m = 100.0,
    potential = V2,

    # initial Gaussian packet
    E0      = 0.4,
    x0_init = -30.0,
    sigma0  = 1.0,

    # exported times (multiples of dt) and the time at which the boosts 1/P are evaluated
    t_export = [240.0, 330.0, 600.0],
    t_boost  = 600.0,

    # quantum state: :stationary (superposition of scattering states, Numerov) or
    # :time_dependent (split-operator)
    quantum_method = :stationary,
    n_σp           = 12.0,      # momentum window p0 ± n_σp·ħ/σ0
    contour_K      = :auto,     # shift of the integration line Im p = K·δ; :auto = 8 with a well, else 0
    period_factor  = 4.0,       # period of the packet copies L = period_factor·(x_max − x_min)

    # grid of ψ(x)
    x_min = -150.0,
    x_max =  150.0,
    x_num = 262144,

    # time step of the trajectories and of the split-operator
    dt = 0.1,

    # complex absorbing potential (:time_dependent only)
    abs_width    = 15.0,
    abs_strength = 0.8,

    # exported grid of the heat maps: step dx_export in x, p_stride points of the Wigner p-grid in p
    x_export   = (-60.0, 60.0),
    dx_export  = 0.3,
    p_export   = (-20.0, 20.0),
    p_stride   = 5,
    cell_value = :average,      # value of Wq in a cell: :average or :max_abs

    # grid of the marginals; marginal_x_boost.txt stores marginal_shift_x + marginal_multiply_x·ρ(x)
    dx_marginal         = 0.025,
    dp_marginal         = 0.025,
    marginal_multiply_x = 10.0,
    marginal_shift_x    = -14.5,

    # Wigner transform
    y_max_wigner = 100.0,
    y_num_wigner = 4096,
    dx_wigner    = 0.025,       # coarsest x step of the Wq samples averaged into a cell

    # trajectories
    N_traj              = 4_000_000,
    rng_seed            = 42,
    max_gen             = 50,
    w_min_spawn         = 0.0,
    V_barrier_threshold = 0.05,
    is_fraction         = 0.5,  # fraction of momenta drawn from the importance proposal (0 = off)

    # energy contours E_contour_min + k·E_contour_step, k = 0 … n_E_contours − 1, and that of E0
    n_contour      = 3001,
    E_contour_min  = 0.2,
    E_contour_step = 0.2,
    n_E_contours   = 10,
)

# ============================================================================
# Code
# ============================================================================

# ----------------------------------------------------------------------------
# Exported cells of Wq
# ----------------------------------------------------------------------------

# Average of the fine samples W_fine[k, j] over the exported cells, resolved by region:
# W_cell[i, c, r] collects the samples of cell (i, c) with the weights region_w[k, r].
function cell_average!(W_cell::Array{Float64, 3}, W_fine::Matrix{Float64}, x_sub::Int,
                       j_centers::Vector{Int}, p_stride::Int, region_w::Matrix{Float64})
    half  = p_stride ÷ 2
    w_end = iseven(p_stride) ? 0.5 : 1.0
    fill!(W_cell, 0.0)
    @threads for c in eachindex(j_centers)
        jc = j_centers[c]
        @inbounds for i in axes(W_cell, 1)
            for k in (i - 1) * x_sub + 1 : i * x_sub
                s = 0.0
                for dj in -half:half
                    w  = abs(dj) == half ? w_end : 1.0
                    s += w * W_fine[k, jc + dj]
                end
                s /= x_sub * p_stride
                for r in axes(W_cell, 3)
                    wr = region_w[k, r]
                    wr > 0 && (W_cell[i, c, r] += wr * s)
                end
            end
        end
    end
    return W_cell
end

# Sample of the largest |W| in each cell, separately for the samples of each region
function cell_maxabs!(W_cell::Array{Float64, 3}, W_fine::Matrix{Float64}, x_sub::Int,
                      j_centers::Vector{Int}, p_stride::Int, region_f::Vector{Int})
    half = p_stride ÷ 2
    fill!(W_cell, 0.0)
    @threads for c in eachindex(j_centers)
        jc = j_centers[c]
        @inbounds for i in axes(W_cell, 1)
            for k in (i - 1) * x_sub + 1 : i * x_sub
                r    = region_f[k]
                best = W_cell[i, c, r]
                for dj in -half:half
                    w = W_fine[k, jc + dj]
                    abs(w) > abs(best) && (best = w)
                end
                W_cell[i, c, r] = best
            end
        end
    end
    return W_cell
end

# Sum of the region parts, each multiplied by its factor
function combine_regions(parts::AbstractArray{Float64, 3}, factors::Vector{Float64})
    W = zeros(Float64, size(parts, 1), size(parts, 2))
    for r in eachindex(factors)
        @views W .+= factors[r] .* parts[:, :, r]
    end
    return W
end

# The region part of the largest |factor · W| (for cell_value = :max_abs)
function combine_regions_maxabs(parts::AbstractArray{Float64, 3}, factors::Vector{Float64})
    W = zeros(Float64, size(parts, 1), size(parts, 2))
    @inbounds for j in axes(parts, 2), i in axes(parts, 1)
        best = 0.0
        for r in eachindex(factors)
            v = factors[r] * parts[i, j, r]
            abs(v) > abs(best) && (best = v)
        end
        W[i, j] = best
    end
    return W
end

# |ψ(x)|² in the points xs (linear interpolation of ψ)
function position_density!(ρ::Vector{Float64}, ψ::Vector{ComplexF64}, x::AbstractVector{Float64},
                           xs::Vector{Float64})
    x0 = x[1]
    dx = x[2] - x[1]
    n  = length(x)
    @inbounds for i in eachindex(xs)
        t = (xs[i] - x0) / dx
        k = clamp(floor(Int, t) + 1, 1, n - 1)
        f = t - (k - 1)
        ρ[i] = abs2((1 - f) * ψ[k] + f * ψ[k + 1])
    end
    return ρ
end

# |ψ̃(p)|², ψ̃(p) = (2πħ)^(-1/2) ∫ ψ(x) e^{-ipx/ħ} dx, by direct summation in the points p
function momentum_density!(ρ::Vector{Float64}, ψ::Vector{ComplexF64}, x::AbstractVector{Float64},
                           p::Vector{Float64}, ħ::Float64)
    dx = x[2] - x[1]
    @threads for j in eachindex(p)
        k = p[j] / ħ
        s = zero(ComplexF64)
        @inbounds for n in eachindex(x)
            s += ψ[n] * cis(-k * x[n])
        end
        ρ[j] = abs2(s) * dx^2 / (2π * ħ)
    end
    return ρ
end

# ----------------------------------------------------------------------------
# ψ(x, t) from stationary scattering states
# ----------------------------------------------------------------------------

# Scattering state on the Numerov grid, matched to the exact plane waves outside:
# e^{ikx} + r e^{−ikx} on the left, t e^{ikx} on the right, k = p/ħ. Returns (k, r, t).
function scattering_state!(ψ::Vector{ComplexF64}, p::ComplexF64, V_in::Vector{Float64},
                           x1::Float64, h::Float64, m::Float64, ħ::Float64)
    t, r = numerov_scattering!(ψ, p, V_in, x1, h, m, ħ)
    k    = p / ħ
    dk   = numerov_wavenumber(k, h) - k
    xn   = x1 + (length(V_in) - 1) * h
    ψ  .*= cis(-dk * x1)
    return k, r * cis(-2 * dk * x1), t * cis(dk * (xn - x1))
end

# Ψ[:, f] = ψ(x, t_frames[f]) = (2πħ)^(-1/2) ∫ dp ψ̃₀(p) ψ_p(x) e^{−ip²t/(2mħ)}: sum with the step
# δ = 2πħ/L, L = period_factor·(x[end] − x[1]), over the line Im p = K·δ and the window
# p0 ± n_σp·ħ/σ0. The Numerov states are used where |V| > V_rel·max|V|, plane waves elsewhere.
function stationary_psi_frames(x::AbstractVector{Float64}, pot, t_frames::Vector{Float64},
                               x0::Float64, p0::Float64, sigma0::Float64, m::Float64, ħ::Float64;
                               n_σp::Float64 = 12.0, K::Float64 = 8.0,
                               period_factor::Float64 = 4.0, V_rel::Float64 = 1e-18)
    nx = length(x)
    dx = x[2] - x[1]
    nf = length(t_frames)

    Vx     = [pot(xv) for xv in x]
    V_cut  = V_rel * maximum(abs, Vx)
    inside = findall(v -> abs(v) > V_cut, Vx)
    iL     = first(inside) - 2
    iR     = last(inside) + 2
    (iL ≥ 1 && iR ≤ nx) || error("The potential is not negligible at the edge of the grid – widen [x_min, x_max].")
    V_in = Vx[iL:iR]
    n_in = length(V_in)

    L    = period_factor * (x[end] - x[1])
    δ    = 2π * ħ / L
    η    = K * δ
    σ_p  = ħ / sigma0
    p_lo = max(p0 - n_σp * σ_p, 10 * η, δ)
    p_hi = p0 + n_σp * σ_p

    # the edge of the window must lie below e^{−2πK−20} (at most e^{−30}) of the maximum of ψ̃₀
    edge_exp = max(2π * K + 20, 30.0)
    n_σp^2 / 2 < edge_exp &&
        @warn "The momentum window is too narrow for the shifted contour – increase n_σp or decrease K." n_σp K
    p_lo > p0 - n_σp * σ_p && (p_lo - p0)^2 / (2 * σ_p^2) < edge_exp &&
        error("The packet has non-negligible components with small or negative p (p0 = $p0, " *
              "σ_p = ħ/σ0 = $σ_p), which the stationary method does not support – increase E0 " *
              "or σ0, or use quantum_method = :time_dependent.")
    p_re = collect(p_lo:δ:p_hi)
    Np   = length(p_re)
    p_c  = p_re .+ im * η

    # the future copy of the packet (amplified by e^{2πK}) must stay left of the grid
    t_max = maximum(t_frames)
    x0 - L + p_hi * t_max / m + 15 * free_width(sigma0, t_max, m, ħ) < x[1] ||
        error("The period L = $L is too short for t = $t_max – increase period_factor.")

    norm_a = (sigma0^2 / (π * ħ^2))^0.25
    ψ̃₀(p) = norm_a * exp(-sigma0^2 * (p - p0)^2 / (2 * ħ^2) - im * p * x0 / ħ)
    C = [δ * ψ̃₀(p_c[k]) * exp(-im * p_c[k]^2 * t_frames[f] / (2 * m * ħ)) / sqrt(2π * ħ)
         for f in 1:nf, k in 1:Np]

    # inner interval: Numerov state of every p, accumulated per task
    k_c = Vector{ComplexF64}(undef, Np)
    r_c = Vector{ComplexF64}(undef, Np)
    t_c = Vector{ComplexF64}(undef, Np)
    ntask   = nthreads()
    acc     = [zeros(ComplexF64, n_in, nf) for _ in 1:ntask]
    chunk_p = cld(Np, ntask)
    @sync for c in 1:ntask
        Threads.@spawn begin
            buf = Vector{ComplexF64}(undef, n_in)
            A   = acc[c]
            for k in (c - 1) * chunk_p + 1 : min(c * chunk_p, Np)
                k_c[k], r_c[k], t_c[k] = scattering_state!(buf, p_c[k], V_in, x[iL], dx, m, ħ)
                @inbounds for f in 1:nf
                    cf = C[f, k]
                    @simd for j in 1:n_in
                        A[j, f] += cf * buf[j]
                    end
                end
            end
        end
    end

    Ψ = zeros(ComplexF64, nx, nf)
    for A in acc
        @views Ψ[iL:iR, :] .+= A
    end

    # outer intervals: e^{ikx} + r e^{−ikx} on the left, t e^{ikx} on the right
    outer   = vcat(1:iL-1, iR+1:nx)
    chunk_x = cld(length(outer), ntask)
    @sync for c in 1:ntask
        Threads.@spawn begin
            s = Vector{ComplexF64}(undef, nf)
            @inbounds for jj in (c - 1) * chunk_x + 1 : min(c * chunk_x, length(outer))
                j  = outer[jj]
                xj = x[j]
                fill!(s, 0)
                if j < iL
                    for k in 1:Np
                        e   = cis(k_c[k] * xj)
                        amp = e + r_c[k] / e
                        for f in 1:nf
                            s[f] += C[f, k] * amp
                        end
                    end
                else
                    for k in 1:Np
                        amp = t_c[k] * cis(k_c[k] * xj)
                        for f in 1:nf
                            s[f] += C[f, k] * amp
                        end
                    end
                end
                for f in 1:nf
                    Ψ[j, f] = s[f]
                end
            end
        end
    end

    at   = [abs(C[1, k] * t_c[k]) for k in 1:Np]
    info = (Np = Np, δ = δ, η = η, p_lo = p_lo, p_hi = p_hi, x_L = x[iL], x_R = x[iR], n_in = n_in,
            norm_window = sum(abs2(ψ̃₀(p)) for p in p_re) * δ, edge_hi = at[end] / maximum(at))
    return Ψ, info
end

# ----------------------------------------------------------------------------
# Potential maxima, energy contours
# ----------------------------------------------------------------------------

# Positions of the maxima of V on [x_lo, x_hi] (sign changes of V′ from + to −)
function potential_maxima(pot, x_lo::Float64, x_hi::Float64; scan_step::Float64 = 0.01,
                          tol::Float64 = 1e-12)
    maxima = Float64[]
    for k in 0:ceil(Int, (x_hi - x_lo) / scan_step) - 1
        a = x_lo + k * scan_step
        b = min(a + scan_step, x_hi)
        (pot.dV(a) > 0 && pot.dV(b) ≤ 0) || continue
        while b - a > tol
            mid = 0.5 * (a + b)
            if pot.dV(mid) > 0
                a = mid
            else
                b = mid
            end
        end
        push!(maxima, round(0.5 * (a + b); digits = 9) + 0.0)
    end
    return maxima
end

# Contour H(x, p) = p²/2m + V(x) = E as a polyline (x, p) over the grid xg; every classically
# allowed region gives one line closed through its turning points, lines are separated by NaN.
function energy_contour(pot, E::Float64, m::Float64, xg::AbstractVector{Float64}; tol::Float64 = 1e-12)
    n  = length(xg)
    cx = Float64[]
    cp = Float64[]

    function turning_point(x_in, x_out)
        while abs(x_out - x_in) > tol
            mid = 0.5 * (x_in + x_out)
            if pot(mid) < E
                x_in = mid
            else
                x_out = mid
            end
        end
        return 0.5 * (x_in + x_out)
    end

    k = 1
    while k ≤ n
        if pot(xg[k]) ≥ E
            k += 1
            continue
        end
        k1 = k
        while k < n && pot(xg[k + 1]) < E
            k += 1
        end
        k2 = k
        k += 1

        left_tp  = k1 > 1
        right_tp = k2 < n
        bx = collect(xg[k1:k2])
        bp = [sqrt(2 * m * (E - pot(xv))) for xv in bx]
        if left_tp
            pushfirst!(bx, turning_point(xg[k1], xg[k1 - 1]))
            pushfirst!(bp, 0.0)
        end
        if right_tp
            push!(bx, turning_point(xg[k2], xg[k2 + 1]))
            push!(bp, 0.0)
        end

        isempty(cx) || (push!(cx, NaN); push!(cp, NaN))
        if right_tp
            append!(cx, bx);                  append!(cp, bp)
            append!(cx, reverse(bx)[2:end]);  append!(cp, -reverse(bp)[2:end])
        elseif left_tp
            append!(cx, reverse(bx));         append!(cp, -reverse(bp))
            append!(cx, bx[2:end]);           append!(cp, bp[2:end])
        else
            append!(cx, bx);                  append!(cp, bp)
            push!(cx, NaN);                   push!(cp, NaN)
            append!(cx, bx);                  append!(cp, -bp)
        end
    end
    return cx, cp
end

# ----------------------------------------------------------------------------
# Semiclassical histograms
# ----------------------------------------------------------------------------

# Add the branches to the histogram W[ix, ip, r + 1] of Wc, to the marginals rx[ix, r + 1], rp[ip]
# (own grids, all branches) and to the weights P[r] of the regions r ≥ 1 (r = number of potential
# maxima left of the branch). Returns the total weight.
function bin_branches!(W::Array{Float64, 3}, rx::Matrix{Float64}, rp::Vector{Float64},
                       P::AbstractVector{Float64}, b::Branches, x_bounds::Vector{Float64},
                       x_edge::Float64, dx::Float64, p_edge::Float64, dp::Float64,
                       xm_edge::Float64, dx_m::Float64, pm_edge::Float64, dp_m::Float64)
    x_n, p_n = size(W, 1), size(W, 2)
    xm_n, pm_n = size(rx, 1), length(rp)
    w_all = 0.0
    @inbounds for k in eachindex(b.x)
        xk = b.x[k]; pk = b.p[k]; wk = b.w[k]
        r  = searchsortedlast(x_bounds, xk)
        ix = floor(Int, (xk - x_edge) / dx) + 1
        ip = floor(Int, (pk - p_edge) / dp) + 1
        (1 ≤ ix ≤ x_n && 1 ≤ ip ≤ p_n) && (W[ix, ip, r + 1] += wk)
        ixm = floor(Int, (xk - xm_edge) / dx_m) + 1
        ipm = floor(Int, (pk - pm_edge) / dp_m) + 1
        1 ≤ ixm ≤ xm_n && (rx[ixm, r + 1] += wk)
        1 ≤ ipm ≤ pm_n && (rp[ipm] += wk)
        w_all += wk
        r > 0 && (P[r] += wk)
    end
    return w_all
end

# Initial trajectories; with is_fraction > 0 the last α·N momenta are redrawn from the importance
# proposal and all trajectories get the weights ρ/q normalised to mean 1.
function sample_trajectories(par, p0::Float64)
    (; ħ, m, potential, x0_init, sigma0, N_traj, is_fraction) = par
    σ_x, σ_p = wigner_widths(sigma0, ħ)
    rng  = Xoshiro(par.rng_seed)
    x, p = sample_packet(rng, N_traj, x0_init, p0, sigma0, ħ)
    w    = ones(N_traj)
    println("\n  initial σ_x = ", round(σ_x; digits = 4), ",  σ_p = ", round(σ_p; digits = 4))
    is_fraction > 0 || return x, p, w

    prop = importance_proposal(potential, x0_init, p0, σ_p, is_fraction, m, ħ, N_traj;
                               x_search_max = par.x_max - x0_init)
    if prop === nothing
        @warn "No momentum passes the barrier – importance sampling is not used."
        return x, p, w
    end
    N_g     = round(Int, is_fraction * N_traj)
    α       = N_g / N_traj
    n_plain = count(>(prop.p_c), p)
    p[N_traj - N_g + 1 : end] .= proposal_sample(rng, prop, N_g)
    w       = importance_weights(p, p0, σ_p, α, prop)
    mean_w  = sum(w) / N_traj
    w     ./= mean_w
    @printf("  importance sampling: α = %.3g, %s proposal with centre p_c = %.4g, width σ_c = %.4g (%.2f σ_p above p0)\n",
            α, prop.tab === nothing ? "Gaussian" : "tabulated ∝ ρ·T", prop.p_c, prop.σ_c, (prop.p_c - p0) / σ_p)
    @printf("  effective number of trajectories: transmitted part %.3g, whole packet %.3g (of %d)\n",
            prop.ess.is_T, prop.ess.is_all, N_traj)
    @printf("  trajectories with p > p_c: %d (%d without importance sampling)\n", count(>(prop.p_c), p), n_plain)
    @printf("  mean weight before normalisation = %.6f, largest weight = %.3f\n", mean_w, maximum(w))
    return x, p, w
end

# ----------------------------------------------------------------------------
function main(par)
    (; ħ, m, potential, E0, x0_init, sigma0, t_export, t_boost, quantum_method, cell_value, dt,
       x_min, x_max, x_num, x_export, dx_export, p_export, p_stride, dx_marginal, dp_marginal,
       marginal_multiply_x, marginal_shift_x, y_num_wigner, N_traj) = par
    name       = "$(potential.name)_E=$(E0)_sigma=$(sigma0)"
    out_folder = "origin_" * name

    quantum_method in (:stationary, :time_dependent) ||
        error("quantum_method must be :stationary or :time_dependent")
    cell_value in (:average, :max_abs) || error("cell_value must be :average or :max_abs")
    0 ≤ par.is_fraction < 1 || error("is_fraction must lie in [0, 1)")

    p0     = packet_momentum(E0, potential, x0_init, m)
    energy = packet_energy(p0, potential, x0_init, m)

    # frames = exported times and the boost time
    t_frames = sort(unique(vcat(t_export, t_boost)))
    steps    = round.(Int, t_frames ./ dt)
    (all(isapprox.(steps .* dt, t_frames; atol = 1e-9)) && allunique(steps)) ||
        error("The times t_export and t_boost must be multiples of dt = $dt.")
    n_t       = length(t_frames)
    is_export = [t in t_export for t in t_frames]
    i_boost   = findfirst(==(t_boost), t_frames)

    x  = LinRange(x_min, x_max, x_num)
    dx = x[2] - x[1]

    # regions for P and the boosts: region r lies between the potential maxima r and r + 1
    x_bounds   = potential_maxima(potential, x_min, x_max)
    n_reg      = length(x_bounds)
    reg_labels = [r < n_reg ? @sprintf("%.4g<x<%.4g", x_bounds[r], x_bounds[r + 1]) :
                              @sprintf("x>%.4g", x_bounds[r]) for r in 1:n_reg]
    i_reg      = vcat([searchsortedfirst(x, b) for b in x_bounds], x_num + 1)

    # exported x grid; Wq is sampled in x_sub points per cell, snapped to nodes of the ψ grid
    x_n   = round(Int, (x_export[2] - x_export[1]) / dx_export) + 1
    x_e   = [x_export[1] + (i - 1) * dx_export for i in 1:x_n]
    x_sub = max(1, ceil(Int, dx_export / par.dx_wigner - 1e-9))
    x_f   = [x[1] + round((x_e[i] + ((k - 0.5) / x_sub - 0.5) * dx_export - x[1]) / dx) * dx
             for i in 1:x_n for k in 1:x_sub]

    # fraction of each sample interval lying in the regions (column r + 1), and the region of its centre
    δ_f       = dx_export / x_sub
    reg_edges = vcat(-Inf, x_bounds, Inf)
    region_w  = [max(min(xv + δ_f / 2, reg_edges[r + 1]) - max(xv - δ_f / 2, reg_edges[r]), 0.0) / δ_f
                 for xv in x_f, r in 1:n_reg + 1]
    region_f  = [searchsortedlast(x_bounds, xv) + 1 for xv in x_f]

    # Wigner transform with dy a multiple of dx (no interpolation of ψ); the exported p cells are
    # centred on every p_stride-th point of its momentum grid, anchored at p = 0
    n_dy   = max(1, round(Int, 2 * par.y_max_wigner / (y_num_wigner * dx)))
    dy_w   = n_dy * dx
    y_w    = [(j - 1 - y_num_wigner ÷ 2) * dy_w for j in 1:y_num_wigner]
    p_w    = π * ħ .* fftshift(fft_frequencies(y_num_wigner, dy_w))
    dp_w   = π * ħ / (y_num_wigner * dy_w)
    j0     = y_num_wigner ÷ 2 + 1
    p_half = p_stride ÷ 2
    j_e    = [j for j in 1 + p_half : y_num_wigner - p_half
              if (j - j0) % p_stride == 0 && p_export[1] ≤ p_w[j] ≤ p_export[2]]
    p_e    = p_w[j_e]
    p_n    = length(p_e)
    dp_e   = p_stride * dp_w
    (p_export[1] < p_w[1] || p_export[2] > p_w[end]) &&
        error("p_export exceeds the range [$(p_w[1]), $(p_w[end])] of the Wigner transform – increase y_num_wigner.")

    # grids of the marginals
    xm_n = round(Int, (x_export[2] - x_export[1]) / dx_marginal) + 1
    x_m  = [x_export[1] + (i - 1) * dx_marginal for i in 1:xm_n]
    pm_n = round(Int, (p_export[2] - p_export[1]) / dp_marginal) + 1
    p_m  = [p_export[1] + (j - 1) * dp_marginal for j in 1:pm_n]
    region_m = [searchsortedlast(x_bounds, xv) for xv in x_m]

    mkpath(out_folder)
    println("=== Export of Wq, Wc, marginals and contours for Origin ===")
    println("  potential:  ", potential.name, ",  E0 = ", energy)
    println("  times:      ", t_export, ",  boost from t = ", t_boost)
    println("  V maxima:   ", x_bounds, "  →  regions: ", join(reg_labels, ",  "))
    @printf("  heat maps:  x ∈ [%.4g, %.4g] step %.4g (%d points), p ∈ [%.4g, %.4g] step %.4g (%d points)\n",
            x_e[1], x_e[end], dx_export, x_n, p_e[1], p_e[end], dp_e, p_n)
    @printf("  Wq samples: %d × %d per cell (step %.4g in x, %.4g in p), cell value: %s\n",
            x_sub, p_stride, dx_export / x_sub, dp_w, cell_value)
    @printf("  marginals:  %d points in x, %d points in p\n", xm_n, pm_n)
    println("  quantum:    ", quantum_method, ",  trajectories: ", N_traj, ",  threads: ", nthreads())

    # ---------- quantum part ----------
    workspaces = [WignerWorkspace(y_num_wigner) for _ in 1:nthreads()]
    W_full     = Matrix{Float64}(undef, x_n * x_sub, y_num_wigner)
    Wq_frames  = [Array{Float64, 3}(undef, x_n, p_n, n_reg + 1) for _ in 1:n_t]
    rho_qx     = [Vector{Float64}(undef, xm_n) for _ in 1:n_t]
    rho_qp     = [Vector{Float64}(undef, pm_n) for _ in 1:n_t]
    norm_q     = Vector{Float64}(undef, n_t)
    P_q        = Matrix{Float64}(undef, n_reg, n_t)

    function quantum_frame!(i_t::Int, ψ::Vector{ComplexF64})
        wigner!(W_full, ψ, x, x_f, y_w, workspaces; ħ = ħ)
        if cell_value == :max_abs
            cell_maxabs!(Wq_frames[i_t], W_full, x_sub, j_e, p_stride, region_f)
        else
            cell_average!(Wq_frames[i_t], W_full, x_sub, j_e, p_stride, region_w)
        end
        position_density!(rho_qx[i_t], ψ, x, x_m)
        momentum_density!(rho_qp[i_t], ψ, x, p_m, ħ)
        norm_q[i_t] = sum(abs2, ψ) * dx
        for r in 1:n_reg
            P_q[r, i_t] = sum(abs2, @view ψ[i_reg[r]:i_reg[r + 1] - 1]) * dx
        end
        return nothing
    end

    if quantum_method == :stationary
        K = par.contour_K == :auto ? (n_reg ≥ 2 ? 8.0 : 0.0) : Float64(par.contour_K)
        println("\nQuantum state from stationary scattering states, contour shift K = ", K)
        @time begin
            Ψ, info = stationary_psi_frames(x, potential, t_frames, x0_init, p0, sigma0, m, ħ;
                                            n_σp = par.n_σp, K = K, period_factor = par.period_factor)
            @printf("  %d momenta p ∈ [%.4g, %.4g] + i·%.3g (step %.3g), Numerov on x ∈ [%.4g, %.4g] (%d points)\n",
                    info.Np, info.p_lo, info.p_hi, info.η, info.δ, info.x_L, info.x_R, info.n_in)
            @printf("  norm of ψ̃₀ in the momentum window: %.15f\n", info.norm_window)
            info.edge_hi > 1e-12 &&
                @warn "The transmitted wave is not negligible at the upper edge of the momentum window – increase n_σp." info.edge_hi
            @showprogress desc = "Quantum: " for i_t in 1:n_t
                quantum_frame!(i_t, Ψ[:, i_t])
            end
        end
    else
        println("\nQuantum state by the split-operator method")
        S = SplitOperator(x, potential.(x) .+ absorbing_potential(x, par.abs_width, par.abs_strength), dt, m, ħ)
        ψ = gaussian_packet.(x, x0_init, p0, sigma0, ħ)
        frame_of_step = Dict(s => i_t for (i_t, s) in enumerate(steps))
        @time @showprogress desc = "Quantum: " for i in 0:maximum(steps)
            i > 0 && split_step!(ψ, S)
            haskey(frame_of_step, i) && quantum_frame!(frame_of_step[i], ψ)
        end
    end

    # ---------- semiclassical part ----------
    xs, ps, ws = sample_trajectories(par, p0)
    chunks     = chunk_branches(xs, ps, ws, nthreads())
    Wc_frames  = [zeros(Float64, x_n, p_n, n_reg + 1) for _ in 1:n_t]
    rho_cx     = [zeros(Float64, xm_n, n_reg + 1) for _ in 1:n_t]
    rho_cp     = [zeros(Float64, pm_n) for _ in 1:n_t]
    norm_c     = zeros(Float64, n_t)
    P_c        = zeros(Float64, n_reg, n_t)

    println("\nTrajectories with tunnelling…")
    @time for i_t in 1:n_t
        advance!(chunks, steps[i_t], potential, dt, m, ħ; V_threshold = par.V_barrier_threshold,
                 w_min_spawn = par.w_min_spawn, max_gen = par.max_gen)
        for b in chunks
            norm_c[i_t] += bin_branches!(Wc_frames[i_t], rho_cx[i_t], rho_cp[i_t], view(P_c, :, i_t), b,
                                         x_bounds, x_e[1] - dx_export / 2, dx_export, p_e[1] - dp_e / 2, dp_e,
                                         x_m[1] - dx_marginal / 2, dx_marginal, p_m[1] - dp_marginal / 2, dp_marginal)
        end
        Wc_frames[i_t] .*= 1.0 / (N_traj * dx_export * dp_e)
        rho_cx[i_t]    .*= 1.0 / (N_traj * dx_marginal)
        rho_cp[i_t]    .*= 1.0 / (N_traj * dp_marginal)
    end
    norm_c ./= N_traj
    P_c    ./= N_traj
    n_branches = sum(length, chunks)
    gen_max    = maximum(b -> maximum(b.gen), chunks)
    println("  branches: ", n_branches, " (", n_branches - N_traj, " reflected), deepest generation: ", gen_max)
    gen_max ≥ par.max_gen && @warn "max_gen reached – later reflections are not followed." par.max_gen

    # boost of each region = 1/(its probability at t_boost), separately for Wq and Wc
    boost_q = 1.0 ./ P_q[:, i_boost]
    boost_c = 1.0 ./ P_c[:, i_boost]
    combine_q     = cell_value == :max_abs ? combine_regions_maxabs : combine_regions
    no_boost      = ones(n_reg + 1)
    factors_q     = vcat(1.0, boost_q)
    factors_c     = vcat(1.0, boost_c)
    write_boost_q = n_reg > 0 && all(isfinite, boost_q)
    write_boost_c = n_reg > 0 && all(isfinite, boost_c)
    n_reg > 0 && !write_boost_q &&
        @warn "The quantum probability of a region vanishes at t_boost – Wq_boost is not written." t_boost boost_q
    n_reg > 0 && !write_boost_c &&
        @warn "The semiclassical probability of a region vanishes at t_boost – Wc_boost is not written." t_boost boost_c

    # ---------- output ----------
    println("\nWriting files…")
    for (i_t, t) in enumerate(t_frames)
        is_export[i_t] || continue
        tag = @sprintf("t=%05.2f", t)
        write_xyz(joinpath(out_folder, "Wq_$(tag).txt"), x_e, p_e, combine_q(Wq_frames[i_t], no_boost), "Wq")
        write_xyz(joinpath(out_folder, "Wc_$(tag).txt"), x_e, p_e, combine_regions(Wc_frames[i_t], no_boost), "Wc")
        write_boost_q && write_xyz(joinpath(out_folder, "Wq_boost_$(tag).txt"), x_e, p_e,
                                   combine_q(Wq_frames[i_t], factors_q), "Wq_boost")
        write_boost_c && write_xyz(joinpath(out_folder, "Wc_boost_$(tag).txt"), x_e, p_e,
                                   combine_regions(Wc_frames[i_t], factors_c), "Wc_boost")
    end

    # marginals, with those of the free packet (not boosted)
    rho_free_x(t) = (σ_t = free_width(sigma0, t, m, ħ);
                     [exp(-(xv - x0_init - p0 * t / m)^2 / σ_t^2) / (sqrt(π) * σ_t) for xv in x_m])
    rho_free_p = [sigma0 / (sqrt(π) * ħ) * exp(-sigma0^2 * (pv - p0)^2 / ħ^2) for pv in p_m]
    scaled(ρ)  = marginal_shift_x .+ marginal_multiply_x * ρ

    hx = ["x"];  cx = [x_m]
    hxb = ["x"]; cxb = [x_m]
    hp = ["p"];  cp = [p_m]
    for (i_t, t) in enumerate(t_frames)
        is_export[i_t] || continue
        q, c, f = (@sprintf("%s(t=%g)", s, t) for s in ("rho_q", "rho_c", "rho_free"))
        push!(hx, q, c, f)
        push!(cx, rho_qx[i_t], rho_cx[i_t] * no_boost, rho_free_x(t))
        write_boost_q && (push!(hxb, q); push!(cxb, scaled(rho_qx[i_t] .* factors_q[region_m .+ 1])))
        write_boost_c && (push!(hxb, c); push!(cxb, scaled(rho_cx[i_t] * factors_c)))
        push!(hxb, f)
        push!(cxb, scaled(rho_free_x(t)))
        push!(hp, q, c)
        push!(cp, rho_qp[i_t], rho_cp[i_t])
    end
    push!(hp, "rho_free")
    push!(cp, rho_free_p)
    write_columns(joinpath(out_folder, "marginal_x.txt"), hx, cx)
    write_columns(joinpath(out_folder, "marginal_p.txt"), hp, cp)
    (write_boost_q || write_boost_c) && write_columns(joinpath(out_folder, "marginal_x_boost.txt"), hxb, cxb)

    # energy contours and the potential
    x_cont    = LinRange(x_export[1], x_export[2], par.n_contour)
    V_cont    = [potential(xv) for xv in x_cont]
    E_visible = max(abs(p_export[1]), abs(p_export[2]))^2 / (2 * m) + maximum(V_cont)
    E_levels  = [par.E_contour_min + k * par.E_contour_step for k in 0:par.n_E_contours-1]
    filter!(E -> !isapprox(E, energy; atol = 1e-9), E_levels)
    E_outside = filter(E -> E > E_visible + 1e-9, E_levels)
    isempty(E_outside) ||
        @warn "The contours of these levels lie outside the window p_export (only levels ≤ $(round(E_visible; digits = 4)) reach it)." E_outside
    hc = String[]; cc = Vector{Float64}[]
    for (label, E) in zip(vcat(@sprintf("E0=%g", energy), [@sprintf("E=%g", E) for E in E_levels]),
                          vcat(energy, E_levels))
        xc, pc = energy_contour(potential, E, m, x_cont)
        push!(hc, "x($label)", "p($label)")
        push!(cc, xc, pc)
    end
    write_columns(joinpath(out_folder, "contours.txt"), hc, cc)
    write_columns(joinpath(out_folder, "potential.txt"), ["x", "V"], [collect(x_cont), V_cont])

    # norms, probabilities of the regions and the boosts
    hs = ["t", "norm_q", "norm_c"]
    cs = [t_frames, norm_q, norm_c]
    for r in 1:n_reg
        push!(hs, "P_q($(reg_labels[r]))", "P_c($(reg_labels[r]))")
        push!(cs, P_q[r, :], P_c[r, :])
    end
    for r in 1:n_reg
        push!(hs, "boost_q($(reg_labels[r]))", "boost_c($(reg_labels[r]))")
        push!(cs, fill(boost_q[r], n_t), fill(boost_c[r], n_t))
    end
    write_columns(joinpath(out_folder, "summary.txt"), hs, cs)

    println("\n      t      norm_q      norm_c")
    for i_t in 1:n_t
        @printf("  %5.2f  %10.6f  %10.6f\n", t_frames[i_t], norm_q[i_t], norm_c[i_t])
    end
    for r in 1:n_reg
        println("\nRegion ", reg_labels[r], ":\n      t         P_q         P_c")
        for i_t in 1:n_t
            @printf("  %5.2f  %10.4e  %10.4e\n", t_frames[i_t], P_q[r, i_t], P_c[r, i_t])
        end
        @printf("  boost (from t = %g):  boost_q = %.6g,  boost_c = %.6g\n", t_boost, boost_q[r], boost_c[r])
    end
    println("\nDone. Data saved in: ", out_folder)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(params)
end
