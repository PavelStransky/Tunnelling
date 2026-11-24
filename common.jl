# Shared library of the 1D tunnelling scripts. Conventions are described in README.md.

using FFTW
using LinearAlgebra
using Printf
using Random
using Statistics
using DelimitedFiles
using ProgressMeter
using Base.Threads

# ============================================================================
# Potentials
# ============================================================================

struct Potential{FV, FD}
    name::Symbol
    V::FV
    dV::FD
end

@inline (pot::Potential)(x) = pot.V(x)

const V0 = Potential(:V0, x -> 0.0, x -> 0.0)

const V1 = Potential(:V1, x -> exp(-x^2), x -> -2 * x * exp(-x^2))

const V2 = Potential(:V2, x -> (1 + x^2) * exp(-x^2), x -> -2 * x^3 * exp(-x^2))

const V3 = Potential(:V3, x -> (0.5 + 2.156 * x^2) * exp(-x^2),
                     x -> (2 * 2.156 * x - 2 * x * (0.5 + 2.156 * x^2)) * exp(-x^2))

const V_double = Potential(:V_double, x -> 0.6 * (exp(-(x - 4.0)^2) + exp(-(x + 4.0)^2)),
                           x -> 0.6 * (-2 * (x - 4.0) * exp(-(x - 4.0)^2) - 2 * (x + 4.0) * exp(-(x + 4.0)^2)))

const V_triple = Potential(:V_triple, x -> 0.6 * (exp(-(x + 7.0)^2) + exp(-x^2) + exp(-(x - 7.0)^2)),
                           x -> 0.6 * (-2 * (x + 7.0) * exp(-(x + 7.0)^2) - 2 * x * exp(-x^2) -
                                       2 * (x - 7.0) * exp(-(x - 7.0)^2)))

const V_asym = Potential(:V_asym, x -> 0.8 * exp(-(x + 5.0)^2) + 0.4 * exp(-(x - 3.0)^2),
                         x -> 0.8 * (-2 * (x + 5.0)) * exp(-(x + 5.0)^2) +
                              0.4 * (-2 * (x - 3.0)) * exp(-(x - 3.0)^2))

# ============================================================================
# Gaussian wave packet
# ============================================================================

gaussian_packet(x, x0, p0, σ0, ħ) =
    (1.0 / (π * σ0^2))^0.25 * exp(-(x - x0)^2 / (2 * σ0^2) + 1im * p0 * (x - x0) / ħ)

packet_momentum(E, pot, x0, m) = sqrt(2 * m * (E - pot(x0)))

packet_energy(p0, pot, x0, m) = p0^2 / (2 * m) + pot(x0)

# Width parameter σ(t) of the freely spreading packet, |ψ|² ∝ exp(−(x − x0 − p0 t/m)²/σ(t)²)
free_width(σ0, t, m, ħ) = σ0 * sqrt(1 + (ħ * t / (m * σ0^2))^2)

# Wigner function of the freely moving packet
free_wigner(x, p, t, x0, p0, σ0, m, ħ) =
    exp(-(x - x0 - p * t / m)^2 / σ0^2 - σ0^2 * (p - p0)^2 / ħ^2) / (π * ħ)

# ============================================================================
# Split-operator propagation
# ============================================================================

# FFT frequencies (cycles per unit length) of n samples with spacing d
function fft_frequencies(n::Int, d::Float64)
    half = (n - 1) ÷ 2 + 1
    return [(k < half ? k : k - n) / (n * d) for k in 0:n-1]
end

# Quadratic complex absorbing potential in layers of the given width at both ends of the grid
function absorbing_potential(x::AbstractVector{Float64}, width::Float64, strength::Float64)
    x_min, x_max = first(x), last(x)
    V = zeros(ComplexF64, length(x))
    @inbounds for i in eachindex(x)
        if x[i] < x_min + width
            s    = (x_min + width - x[i]) / width
            V[i] = -1im * strength * s^2
        elseif x[i] > x_max - width
            s    = (x[i] - (x_max - width)) / width
            V[i] = -1im * strength * s^2
        end
    end
    return V
end

# Phase factors and FFT plans of one time step; one instance can be shared by all tasks
struct SplitOperator{PF, PB}
    expV::Vector{ComplexF64}
    expK::Vector{ComplexF64}
    p::Vector{Float64}
    fft::PF
    ifft::PB
end

function SplitOperator(x::AbstractVector{Float64}, V::AbstractVector, dt::Float64, m::Float64, ħ::Float64)
    n    = length(x)
    p    = ħ .* 2π .* fft_frequencies(n, x[2] - x[1])
    expV = @. exp(-0.5im * V * dt / ħ)
    expK = @. exp(-1.0im * (p^2) * dt / (2 * m * ħ))
    buf  = Vector{ComplexF64}(undef, n)
    return SplitOperator(expV, expK, p, plan_fft!(buf; flags = FFTW.MEASURE),
                         plan_ifft!(buf; flags = FFTW.MEASURE))
end

@inline function split_step!(ψ::Vector{ComplexF64}, S::SplitOperator)
    expV, expK = S.expV, S.expK
    @inbounds @simd for i in eachindex(ψ)
        ψ[i] = expV[i] * ψ[i]
    end
    S.fft * ψ
    @inbounds @simd for i in eachindex(ψ)
        ψ[i] = expK[i] * ψ[i]
    end
    S.ifft * ψ
    @inbounds @simd for i in eachindex(ψ)
        ψ[i] = expV[i] * ψ[i]
    end
    return ψ
end

# Propagate ψ by t_num steps and store it every plot_step steps (snaps[1] = initial state)
function propagate_snapshots!(snaps::Vector{Vector{ComplexF64}}, ψ::Vector{ComplexF64},
                              S::SplitOperator, t_num::Int, plot_step::Int)
    copyto!(snaps[1], ψ)
    for i in 1:t_num
        split_step!(ψ, S)
        i % plot_step == 0 && copyto!(snaps[i ÷ plot_step + 1], ψ)
    end
    return snaps
end

# ⟨a|b⟩ as a plain sum (without the factor dx)
function inner_product(a::Vector{ComplexF64}, b::Vector{ComplexF64})
    re = 0.0
    im = 0.0
    @inbounds @simd for k in eachindex(a)
        a_re = real(a[k]); a_im = imag(a[k])
        b_re = real(b[k]); b_im = imag(b[k])
        re += a_re * b_re + a_im * b_im
        im += a_re * b_im - a_im * b_re
    end
    return re, im
end

# ============================================================================
# Wigner transform
# ============================================================================

# Grid of the Wigner transform: x samples, y samples and the conjugate momentum axis
function wigner_grid(x_left, x_right, x_num::Int, y_max, y_num::Int, ħ)
    dx = (x_right - x_left) / x_num
    dy = 2 * y_max / y_num
    return (x  = [x_left + (i - 1) * dx for i in 1:x_num], dx = dx,
            y  = [-y_max + (j - 1) * dy for j in 1:y_num], dy = dy,
            p  = π * ħ .* fftshift(fft_frequencies(y_num, dy)), dp = π * ħ / (y_num * dy))
end

struct WignerWorkspace{P}
    buf::Vector{ComplexF64}
    plan::P
end

function WignerWorkspace(y_num::Int)
    buf = Vector{ComplexF64}(undef, y_num)
    return WignerWorkspace(buf, plan_fft!(buf; flags = FFTW.MEASURE))
end

# ψ(xq) by linear interpolation on a uniform grid, zero outside
@inline function interpolate_psi(ψ, x1::Float64, dx::Float64, x_end::Float64, xq::Float64)
    (xq < x1 || xq > x_end) && return zero(ComplexF64)
    n = length(ψ)
    t = (xq - x1) / dx
    k = floor(Int, t) + 1
    k >= n && return @inbounds ψ[n]
    f = t - (k - 1)
    return @inbounds (1 - f) * ψ[k] + f * ψ[k + 1]
end

function wigner_rows!(W::Matrix{Float64}, ψ::AbstractVector{ComplexF64}, x_grid, x_samples, y,
                      ws::WignerWorkspace, rows, ħ::Float64)
    n_y   = length(y)
    half  = n_y ÷ 2
    scale = (y[2] - y[1]) / (π * ħ)
    x1    = x_grid[1]
    dx    = x_grid[2] - x_grid[1]
    x_end = x_grid[end]
    buf   = ws.buf
    @inbounds for i in rows
        xc = x_samples[i]
        for j in 1:n_y
            c = conj(interpolate_psi(ψ, x1, dx, x_end, xc - y[j])) *
                interpolate_psi(ψ, x1, dx, x_end, xc + y[j])
            buf[j ≤ half ? j + half : j - half] = c
        end
        ws.plan * buf
        for j in 1:half
            W[i, j]        = real(buf[j + half]) * scale
            W[i, j + half] = real(buf[j])        * scale
        end
    end
    return W
end

# W[i, j] = W(x_samples[i], p[j]) of the state ψ given on x_grid; y, p from wigner_grid.
# One workspace: sequential; a vector of workspaces: one task per workspace.
wigner!(W, ψ, x_grid, x_samples, y, ws::WignerWorkspace; ħ::Float64) =
    wigner_rows!(W, ψ, x_grid, x_samples, y, ws, eachindex(x_samples), ħ)

function wigner!(W, ψ, x_grid, x_samples, y, wss::Vector{<:WignerWorkspace}; ħ::Float64)
    N     = length(x_samples)
    chunk = cld(N, length(wss))
    @sync for (c, ws) in enumerate(wss)
        rows = (c - 1) * chunk + 1 : min(c * chunk, N)
        isempty(rows) || Threads.@spawn wigner_rows!(W, ψ, x_grid, x_samples, y, ws, rows, ħ)
    end
    return W
end

# ============================================================================
# Classical dynamics: truncated Wigner approximation with Gamow tunnelling
# ============================================================================

@inline function verlet_step(x::Float64, p::Float64, dt::Float64, m::Float64, pot)
    p_half = p + 0.5 * dt * (-pot.dV(x))
    x_new  = x + dt * p_half / m
    p_new  = p_half + 0.5 * dt * (-pot.dV(x_new))
    return x_new, p_new
end

# Exit turning point b of the connected forbidden region (V > E) entered at a, in the direction
# of motion, and the highest V met on the way. (NaN, ·) if the region does not end in reach.
function find_barrier_exit(pot, a::Float64, direction::Float64, E::Float64;
                           scan_step::Float64 = 0.05, x_search_max::Float64 = 60.0,
                           tol::Float64 = 1e-10)
    d = direction ≥ 0 ? 1.0 : -1.0
    max_steps = ceil(Int, x_search_max / scan_step)

    # the forbidden region begins right behind a: look for its first point with growing steps,
    # so that a region thinner than scan_step (energy just below a barrier top) is not missed
    x = a
    δ = 1e-7
    while pot(x) ≤ E && δ < scan_step
        x = a + d * δ
        δ *= 2
    end
    steps = 0
    while pot(x) ≤ E && steps < max_steps
        x += d * scan_step
        steps += 1
    end
    pot(x) ≤ E && return (NaN, NaN)

    V_peak    = pot(x)
    x_inside  = x
    x_outside = x
    found     = false
    while steps < max_steps
        x += d * scan_step
        steps += 1
        Vx = pot(x)
        Vx > V_peak && (V_peak = Vx)
        if Vx ≤ E
            x_outside = x
            found = true
            break
        else
            x_inside = x
        end
    end
    found || return (NaN, V_peak)

    lo, hi = x_inside, x_outside
    while abs(hi - lo) > tol
        mid = 0.5 * (lo + hi)
        if pot(mid) > E
            lo = mid
        else
            hi = mid
        end
    end
    return (0.5 * (lo + hi), V_peak)
end

# ∫ √(2m(V − E)) dx between a and b (trapezoidal rule)
function gamow_integral(pot, a::Float64, b::Float64, E::Float64, m::Float64; n_int::Int = 400)
    lo = min(a, b); hi = max(a, b)
    dx = (hi - lo) / (n_int - 1)
    total = 0.0
    for i in 1:n_int
        arg = 2 * m * (pot(lo + (i - 1) * dx) - E)
        val = arg > 0 ? sqrt(arg) : 0.0
        total += ((i == 1 || i == n_int) ? 0.5 : 1.0) * val
    end
    return total * dx
end

# Gamow transmission probability of the first barrier met by a trajectory sent from x_start
# to the right with energy E (1 if there is none)
function first_barrier_T(pot, x_start::Float64, E::Float64, m::Float64, ħ::Float64;
                         scan_step::Float64 = 0.05, x_search_max::Float64 = 60.0,
                         tol::Float64 = 1e-10)
    x = x_start
    max_steps = ceil(Int, x_search_max / scan_step)
    steps = 0
    while pot(x) ≤ E && steps < max_steps
        x += scan_step
        steps += 1
    end
    pot(x) ≤ E && return 1.0
    lo, hi = x - scan_step, x
    while hi - lo > tol
        mid = 0.5 * (lo + hi)
        if pot(mid) > E
            hi = mid
        else
            lo = mid
        end
    end
    a = 0.5 * (lo + hi)

    b, _ = find_barrier_exit(pot, x, 1.0, E; scan_step = scan_step,
                             x_search_max = x_search_max, tol = tol)
    isnan(b) && return 0.0
    return exp(-2 * gamow_integral(pot, a, b, E, m) / ħ)
end

# One Verlet step; at a turning point inside a barrier the trajectory splits. Returns the trunk
# (x, p, w), a flag, and the reflected branch (x_r, p_r, w_r). The trunk of a split is the
# transmitted branch placed at the exit turning point with weight w·T.
@inline function twa_step(x::Float64, p::Float64, w::Float64, pot, dt::Float64, m::Float64,
                          ħ::Float64, V_threshold::Float64)
    x_new, p_new = verlet_step(x, p, dt, m, pot)
    if p * p_new < 0.0 && pot(x_new) > V_threshold
        E         = 0.5 * (p^2 + p_new^2) / (2 * m) + pot(x_new)
        direction = p > 0 ? 1.0 : -1.0
        b, V_peak = find_barrier_exit(pot, x_new, direction, E)
        if !isnan(b) && E < V_peak
            T = exp(-2 * gamow_integral(pot, x_new, b, E, m) / ħ)
            return b, 1e-6 * direction, w * T, true, x_new, p_new, w * (1.0 - T)
        end
    end
    return x_new, p_new, w, false, x_new, p_new, 0.0
end

# Weighted classical branches. gen = number of reflections that created the branch,
# step = time step of the stored state, root = index of the initial trajectory.
struct Branches
    x::Vector{Float64}
    p::Vector{Float64}
    w::Vector{Float64}
    gen::Vector{Int}
    step::Vector{Int}
    root::Vector{Int}
end

Branches(x, p, w, root = collect(1:length(x))) =
    Branches(x, p, w, zeros(Int, length(x)), zeros(Int, length(x)), root)

Base.length(b::Branches) = length(b.x)

# Split the initial trajectories into n contiguous chunks (one per task)
function chunk_branches(x::Vector{Float64}, p::Vector{Float64}, w::Vector{Float64}, n::Int)
    N    = length(x)
    size = cld(N, n)
    return [Branches(x[r], p[r], w[r], collect(r))
            for r in ((c - 1) * size + 1 : min(c * size, N) for c in 1:n) if !isempty(r)]
end

# Advance all branches to time step s_to. Reflected branches are appended and advanced in the
# same call. Outside x_free = (x_L, x_R) the motion is free and is done analytically.
function advance!(b::Branches, s_to::Int, pot, dt::Float64, m::Float64, ħ::Float64;
                  V_threshold::Float64 = 0.05, w_min_spawn::Float64 = 0.0, max_gen::Int = 50,
                  x_free::NTuple{2, Float64} = (-Inf, Inf))
    x_free_L, x_free_R = x_free
    k = 1
    while k ≤ length(b.x)
        x = b.x[k]; p = b.p[k]; w = b.w[k]; g = b.gen[k]
        i = b.step[k]
        while i < s_to
            if x < x_free_L || x > x_free_R
                n_left = s_to - i
                if (x < x_free_L && p ≤ 0.0) || (x > x_free_R && p ≥ 0.0)
                    n_fly = n_left
                else
                    dist  = x < x_free_L ? x_free_L - x : x - x_free_R
                    n_fly = floor(Int, min(Float64(n_left), dist * m / (abs(p) * dt)))
                end
                if n_fly > 0
                    x += n_fly * dt * p / m
                    i += n_fly
                    continue
                end
            end

            i += 1
            x, p, w, split, x_r, p_r, w_r = twa_step(x, p, w, pot, dt, m, ħ, V_threshold)
            if split && w_r > w_min_spawn && g < max_gen
                push!(b.x, x_r); push!(b.p, p_r); push!(b.w, w_r)
                push!(b.gen, g + 1); push!(b.step, i); push!(b.root, b.root[k])
            end
        end
        b.x[k] = x; b.p[k] = p; b.w[k] = w; b.step[k] = s_to
        k += 1
    end
    return b
end

function advance!(chunks::Vector{Branches}, s_to::Int, pot, dt::Float64, m::Float64, ħ::Float64; kw...)
    @sync for b in chunks
        Threads.@spawn advance!(b, s_to, pot, dt, m, ħ; kw...)
    end
    return chunks
end

# Interval outside which |V| and |V′| are below 1e-14 of their maxima on [x_L, x_R]
function free_region(pot, x_L::Float64, x_R::Float64)
    xs = range(x_L, x_R; length = 20_001)
    V  = abs.(pot.(xs))
    dV = abs.(pot.dV.(xs))
    active = (V .> 1e-14 * maximum(V)) .| (dV .> 1e-14 * maximum(dV))
    any(active) || return (0.5 * (x_L + x_R), 0.5 * (x_L + x_R))
    return (xs[max(1, findfirst(active) - 1)], xs[min(length(xs), findlast(active) + 1)])
end

# Weighted histograms of chunked branches (cell i covers edge + (i − 1)·d … edge + i·d)
function histogram!(H::Matrix{Float64}, task_hists::Vector{Matrix{Float64}}, chunks::Vector{Branches},
                    x_edge::Float64, dx::Float64, p_edge::Float64, dp::Float64)
    n_x, n_p = size(H)
    @sync for (c, b) in enumerate(chunks)
        Threads.@spawn begin
            h = task_hists[c]
            fill!(h, 0.0)
            @inbounds for k in eachindex(b.x)
                ix = floor(Int, (b.x[k] - x_edge) / dx) + 1
                ip = floor(Int, (b.p[k] - p_edge) / dp) + 1
                (1 ≤ ix ≤ n_x && 1 ≤ ip ≤ n_p) && (h[ix, ip] += b.w[k])
            end
        end
    end
    fill!(H, 0.0)
    for c in eachindex(chunks)
        H .+= task_hists[c]
    end
    return H
end

function histogram!(H::Vector{Float64}, task_hists::Vector{Vector{Float64}}, chunks::Vector{Branches},
                    x_edge::Float64, dx::Float64)
    n_x = length(H)
    @sync for (c, b) in enumerate(chunks)
        Threads.@spawn begin
            h = task_hists[c]
            fill!(h, 0.0)
            @inbounds for k in eachindex(b.x)
                ix = floor(Int, (b.x[k] - x_edge) / dx) + 1
                1 ≤ ix ≤ n_x && (h[ix] += b.w[k])
            end
        end
    end
    fill!(H, 0.0)
    for c in eachindex(chunks)
        H .+= task_hists[c]
    end
    return H
end

# ----------------------------------------------------------------------------
# Sampling of the initial Wigner function
# ----------------------------------------------------------------------------

# Standard deviations of x and p in the Wigner function of the Gaussian packet
wigner_widths(σ0, ħ) = (σ0 / sqrt(2), ħ / (sqrt(2) * σ0))

function sample_packet(rng::AbstractRNG, N::Int, x0, p0, σ0, ħ)
    σ_x, σ_p = wigner_widths(σ0, ħ)
    x = x0 .+ σ_x .* randn(rng, N)
    p = p0 .+ σ_p .* randn(rng, N)
    return x, p
end

# Proposal g(p) for importance sampling of the initial momenta: momenta are drawn from the mixture
# q = (1 − α)ρ + αg, ρ = N(p0, σ_p), with weights ρ/q. g is the Gaussian N(p_c, σ_c) fitted to
# f = ρ·T (T of the first barrier), or the tabulated envelope of f (tab) if that gives an
# effective sample size of the transmitted part more than tab_gain times larger.
# Returns nothing if no momentum is transmitted.
function importance_proposal(pot, x0::Float64, p0::Float64, σ_p::Float64, α::Float64,
                             m::Float64, ħ::Float64, N::Int;
                             x_search_max::Float64 = 60.0, n_σ::Float64 = 12.0,
                             n_grid::Int = 4001, tab_gain::Float64 = 2.0)
    pg = collect(range(p0 - n_σ * σ_p, p0 + n_σ * σ_p; length = n_grid))
    dp = pg[2] - pg[1]
    ρ  = @. exp(-(pg - p0)^2 / (2 * σ_p^2)) / (sqrt(2π) * σ_p)
    T  = [p > 0 ? first_barrier_T(pot, x0, p^2 / (2 * m) + pot(x0), m, ħ;
                                  x_search_max = x_search_max) : 0.0
          for p in pg]
    f  = ρ .* T
    Σf = sum(f)
    Σf > 0 || return nothing
    p_c = sum(pg .* f) / Σf
    σ_c = sqrt(sum((pg .- p_c) .^ 2 .* f) / Σf)
    σ_c > 0 || return nothing

    I_T = Σf * dp
    ess_of(q) = (is_T   = N * I_T^2 / (sum(ρ .^ 2 .* T .^ 2 ./ q) * dp),
                 is_all = N / (sum(ρ .^ 2 ./ q) * dp))
    q   = @. (1 - α) * ρ + α * exp(-(pg - p_c)^2 / (2 * σ_c^2)) / (sqrt(2π) * σ_c)
    ess = ess_of(q)

    h = [maximum(@view f[max(1, i - 1):min(n_grid, i + 1)]) for i in 1:n_grid]
    C = cumsum(h)
    g = h ./ (C[end] * dp)
    C ./= C[end]
    ess_tab = ess_of(@. (1 - α) * ρ + α * g)
    ess_tab.is_T > tab_gain * ess.is_T || return (p_c = p_c, σ_c = σ_c, ess = ess, tab = nothing)
    return (p_c = p_c, σ_c = σ_c, ess = ess_tab, tab = (p1 = pg[1], dp = dp, g = g, C = C))
end

# Tabulated proposal: momentum for a uniform u ∈ [0, 1), and the density g(p)
@inline function tab_sample(tab, u::Float64)
    i  = searchsortedlast(tab.C, u) + 1
    C0 = i > 1 ? tab.C[i - 1] : 0.0
    return tab.p1 + (i - 1.5 + (u - C0) / (tab.C[i] - C0)) * tab.dp
end

@inline function tab_density(tab, p::Float64)
    i = round(Int, (p - tab.p1) / tab.dp) + 1
    return 1 ≤ i ≤ length(tab.g) ? tab.g[i] : 0.0
end

proposal_sample(rng, prop, n::Int) =
    prop.tab === nothing ? prop.p_c .+ prop.σ_c .* randn(rng, n) :
                           [tab_sample(prop.tab, u) for u in rand(rng, n)]

# g(p) of the proposal
proposal_density(prop, p) =
    prop.tab === nothing ? exp(-(p - prop.p_c)^2 / (2 * prop.σ_c^2)) / (sqrt(2π) * prop.σ_c) :
                           tab_density(prop.tab, p)

# Importance weights ρ/q = 1/((1 − α) + α g/ρ)
function importance_weights(p::Vector{Float64}, p0, σ_p, α, prop)
    if prop.tab === nothing
        p_c, σ_c = prop.p_c, prop.σ_c
        return @. 1 / ((1 - α) + α * (σ_p / σ_c) *
                       exp((p - p0)^2 / (2 * σ_p^2) - (p - p_c)^2 / (2 * σ_c^2)))
    else
        g_ρ = [tab_density(prop.tab, pk) * sqrt(2π) * σ_p * exp((pk - p0)^2 / (2 * σ_p^2)) for pk in p]
        return @. 1 / ((1 - α) + α * g_ρ)
    end
end

# Symmetric importance sampling (N divisible by 4): quadruplets (±δx, ±δp) from ρ and pairs
# (±δx, p) from g; prop = nothing samples ρ only. Weights are normalised to mean 1.
# Returns (x, p, w, c, α), c = mean weight before the normalisation, α = realised fraction from g.
function sample_symmetric(rng::AbstractRNG, N::Int, x0, p0, σ0, ħ, prop, α_target)
    σ_x, σ_p = wigner_widths(σ0, ħ)
    n_base = N ÷ 4
    n_g    = prop === nothing ? 0 : round(Int, α_target * n_base)
    n_ρ    = n_base - n_g
    α      = n_g / n_base

    δx   = σ_x .* randn(rng, n_ρ)
    δp   = σ_p .* randn(rng, n_ρ)
    δx_g = σ_x .* randn(rng, 2 * n_g)
    p_g  = prop === nothing ? Float64[] : proposal_sample(rng, prop, 2 * n_g)

    x = x0 .+ vcat(δx, δx, -δx, -δx, δx_g, -δx_g)
    p = vcat(p0 .+ δp, p0 .- δp, p0 .+ δp, p0 .- δp, p_g, p_g)
    n_g > 0 || return x, p, ones(Float64, N), 1.0, α
    w = importance_weights(p, p0, σ_p, α, prop)
    c = sum(w) / N
    w ./= c
    return x, p, w, c, α
end

# ============================================================================
# Stationary scattering states (Numerov)
# ============================================================================

# Wavenumber of the free discrete Numerov wave with step h
numerov_wavenumber(k, h) = 2 / h * asin(k * h / 2 / sqrt(1 + (k * h)^2 / 12))

# Scattering state of a wave incident from the left with momentum p (real or complex), integrated
# from right to left on the grid x_L + (j − 1)h, V[j] = V(x_j). Returns (t, r) relative to the
# discrete free waves e^{±ik̃x}. If χ is a vector, the state normalised to a unit incident wave is
# stored in the nodes x_L + (i − 1)·n_sub·h.
function numerov_scattering!(χ, p::Number, V::Vector{Float64}, x_L::Float64, h::Float64,
                             m::Float64, ħ::Float64; n_sub::Int = 1)
    n   = length(V)
    k2  = (p / ħ)^2
    c   = h^2 / 12
    c2m = 2 * m / ħ^2
    k̃   = numerov_wavenumber(p / ħ, h)

    ψ2  = cis(k̃ * (x_L + (n - 1) * h))
    ψ1  = cis(k̃ * (x_L + (n - 2) * h))
    g2  = 1 + c * (k2 - c2m * V[n])
    kk1 = k2 - c2m * V[n - 1]
    g1  = 1 + c * kk1
    log_scale = 0.0

    rec = χ !== nothing
    ls  = rec ? zeros(Float64, length(χ)) : Float64[]
    if rec
        χ[end] = ψ2
        n_sub == 1 && (χ[end - 1] = ψ1)
    end

    @inbounds for j in n-2:-1:1
        kk0 = k2 - c2m * V[j]
        g0  = 1 + c * kk0
        ψ0  = (2 * (1 - 5 * c * kk1) * ψ1 - g2 * ψ2) / g0

        ψ2, ψ1      = ψ1, ψ0
        g2, g1, kk1 = g1, g0, kk0

        a = abs(ψ1)
        if a > 1e100
            ψ1 /= a; ψ2 /= a
            log_scale += log(a)
        end
        if rec && (j - 1) % n_sub == 0
            i = (j - 1) ÷ n_sub + 1
            χ[i]  = ψ1
            ls[i] = log_scale
        end
    end

    x2 = x_L + h
    D  = -2im * sin(k̃ * h)
    A  = (ψ1 * cis(-k̃ * x2) - ψ2 * cis(-k̃ * x_L)) / D
    B  = (ψ2 * cis(k̃ * x_L) - ψ1 * cis(k̃ * x2)) / D

    if rec
        @inbounds for i in eachindex(χ)
            χ[i] = χ[i] * exp(ls[i] - log_scale) / A
        end
    end
    return exp(-log_scale) / A, B / A
end

transmission_amplitude(p, V, x_L, h, m, ħ) = numerov_scattering!(nothing, p, V, x_L, h, m, ħ)[1]

# Narrow resonances of t(p) on the grid p_st (values t_st): phase steps that differ from the local
# median by more than Δφ_flag are zoomed in until the largest step is below Δφ_res.
# Returns [(p_r, Γ_p)], Γ_p = FWHM in p.
function find_narrow_resonances(p_st::AbstractRange{Float64}, t_st::Vector{ComplexF64},
                                V::Vector{Float64}, x_L::Float64, h::Float64, m::Float64, ħ::Float64;
                                Δφ_flag::Float64 = 0.3, Δφ_res::Float64 = 0.05,
                                n_med::Int = 5, n_zoom::Int = 201, max_zoom::Int = 30)
    n  = length(p_st)
    dφ = [angle(t_st[i + 1] * conj(t_st[i])) for i in 1:n-1]
    flagged = [abs(dφ[i] - median(@view dφ[max(1, i - n_med):min(n - 1, i + n_med)])) > Δφ_flag
               for i in 1:n-1]

    resonances = NTuple{2, Float64}[]
    i = 1
    while i ≤ n - 1
        if !flagged[i]
            i += 1
            continue
        end
        j = i
        while j < n - 1 && flagged[j + 1]
            j += 1
        end
        a = p_st[max(1, i - 1)]
        b = p_st[min(n, j + 2)]
        i = j + 1

        found = false
        for _ in 1:max_zoom
            ps = range(a, b; length = n_zoom)
            step(ps) > 4 * eps(b) || break
            ts    = [transmission_amplitude(p, V, x_L, h, m, ħ) for p in ps]
            steps = [abs(angle(ts[k + 1] * conj(ts[k]))) for k in 1:n_zoom-1]
            k     = argmax(steps)
            if steps[k] < Δφ_res
                p_r = ps[k] + step(ps) / 2
                Γ_p = 2 * step(ps) / steps[k]
                any(r -> abs(r[1] - p_r) < r[2] + Γ_p, resonances) || push!(resonances, (p_r, Γ_p))
                found = true
                break
            end
            a = ps[max(1, k - 1)]
            b = ps[min(n_zoom, k + 2)]
        end
        found || println("  WARNING: the phase jump of t(p) near p ≈ ", a, " could not be resolved")
    end
    return resonances
end

# ============================================================================
# Utilities
# ============================================================================

# Maximum of the samples y on the uniform axis t, refined by a parabola through three points
function refined_maximum(y::AbstractVector{Float64}, t::AbstractVector{Float64})
    i = 1
    @inbounds for j in 2:length(y)
        y[j] > y[i] && (i = j)
    end
    if 1 < i < length(y)
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-30
            δ = clamp(0.5 * (y0 - y2) / denom, -1.0, 1.0)
            return t[i] + δ * (t[2] - t[1]), y1 - 0.25 * (y0 - y2) * δ
        end
    end
    return t[i], y[i]
end

# Ridges of the two-time overlap O[i1, i2] = O(t1, t2): the best t2 for each t1 and vice versa
function overlap_ridges(O::Matrix{Float64}, t_axis::Vector{Float64})
    n = length(t_axis)
    t2_max = zeros(n); O_row = zeros(n)
    t1_max = zeros(n); O_col = zeros(n)
    for i in 1:n
        t2_max[i], O_row[i] = refined_maximum(view(O, i, :), t_axis)
        t1_max[i], O_col[i] = refined_maximum(view(O, :, i), t_axis)
    end
    return (t2_max = t2_max, O_row = O_row, t1_max = t1_max, O_col = O_col)
end

# Run work(i, slot) for i = 1…n on nthreads() tasks with dynamic scheduling;
# slot = 1…nthreads() identifies the task (index of its workspace).
function parallel_sweep(work, n::Int; desc::String = "Energy sweep: ")
    progress = Progress(n; desc = desc, showspeed = true)
    lk       = ReentrantLock()
    next_i   = Threads.Atomic{Int}(1)
    @sync for slot in 1:nthreads()
        Threads.@spawn while true
            i = Threads.atomic_add!(next_i, 1)
            i > n && break
            work(i, slot)
            lock(() -> next!(progress), lk)
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Text output (tab-separated, for Origin)
# ----------------------------------------------------------------------------

# Columns of possibly different lengths with a header; NaN and missing rows give empty cells
function write_columns(path::String, header::Vector{String}, cols::Vector{Vector{Float64}})
    open(path, "w") do io
        println(io, join(header, '\t'))
        for r in 1:maximum(length, cols)
            for (c, col) in enumerate(cols)
                c > 1 && print(io, '\t')
                (r ≤ length(col) && !isnan(col[r])) && @printf(io, "%.8g", col[r] + 0.0)
            end
            println(io)
        end
    end
end

# Heat map in the long format: one row (x, p, W) per grid point
function write_xyz(path::String, x::Vector{Float64}, p::Vector{Float64}, W::Matrix{Float64},
                   zname::String)
    open(path, "w") do io
        println(io, "x\tp\t", zname)
        for i in eachindex(x), j in eachindex(p)
            @printf(io, "%.8g\t%.8g\t%.6g\n", x[i], p[j], W[i, j])
        end
    end
end

# Matrix with labelled axes: first row = column axis, first column = row axis
function write_labelled_matrix(path::String, M::Matrix{Float64}, rows, cols, corner::String;
                               fmt::String = "%.6e")
    f = Printf.Format(fmt)
    open(path, "w") do io
        print(io, corner)
        for c in cols
            @printf(io, "\t%.4f", c)
        end
        println(io)
        for (i, r) in enumerate(rows)
            @printf(io, "%.4f", r)
            for j in eachindex(cols)
                print(io, '\t', Printf.format(f, M[i, j]))
            end
            println(io)
        end
    end
end

# Maps M[i_E, i_t] of an energy sweep in the long format: time, energy, one column per map
function write_sweep_xyz(path::String, names::Vector{String}, t_axis, energies, maps)
    open(path, "w") do io
        println(io, "# time\tenergy\t", join(names, '\t'))
        for ie in eachindex(energies), it in eachindex(t_axis)
            @printf(io, "%.4f\t%.4f", t_axis[it], energies[ie])
            for M in maps
                @printf(io, "\t%.8e", M[ie, it])
            end
            println(io)
        end
    end
end

# Two-time overlap matrix, its diagonal and its ridges
function write_overlap_matrix(folder::String, O::Matrix{Float64}, t_axis::Vector{Float64}, ridges)
    write_labelled_matrix(joinpath(folder, "overlap_origin.txt"), O, t_axis, t_axis, "t1\\t2"; fmt = "%.8e")
    writedlm(joinpath(folder, "overlap_data.txt"), O)
    writedlm(joinpath(folder, "time_axis.txt"), t_axis)
    writedlm(joinpath(folder, "overlap_diagonal.txt"), [t_axis diag(O)])
    open(joinpath(folder, "overlap_xyz.txt"), "w") do io
        println(io, "# t1\tt2\toverlap")
        for i1 in eachindex(t_axis), i2 in eachindex(t_axis)
            @printf(io, "%.4f\t%.4f\t%.8e\n", t_axis[i1], t_axis[i2], O[i1, i2])
        end
    end
    for (file, head, t_max, O_max) in (("t2_max.txt", "# t1\tt2_max\tt2_max-t1\tO_at_max", ridges.t2_max, ridges.O_row),
                                       ("t1_max.txt", "# t2\tt1_max\tt1_max-t2\tO_at_max", ridges.t1_max, ridges.O_col))
        open(joinpath(folder, file), "w") do io
            println(io, head)
            for i in eachindex(t_axis)
                @printf(io, "%.4f\t%.6f\t%.6f\t%.8e\n", t_axis[i], t_max[i], t_max[i] - t_axis[i], O_max[i])
            end
        end
    end
end

# Lines "key<TAB>value[<TAB>value…]"
function write_parameters(path::String, entries)
    open(path, "w") do io
        for (key, value) in entries
            println(io, key, '\t', value isa Union{Tuple, AbstractVector} ? join(value, '\t') : value)
        end
    end
end

# ----------------------------------------------------------------------------
# Plot helpers (the calling script must load Plots)
# ----------------------------------------------------------------------------

const CMAP_SEQUENTIAL = [:black, :navy, :blue, :cyan, :yellow, :orange, :red, :white]
const CMAP_DIVERGING  = [:navy, :blue, :cyan, :white, :yellow, :orange, :red]

# Value below which lies the fraction q of the entries of M
quantile_value(M, q) = (v = sort(vec(M)); v[clamp(round(Int, q * length(v)), 1, length(v))])

function save_heatmap(path::String, x, y, M; colors = CMAP_SEQUENTIAL, show::Bool = false, kw...)
    p = heatmap(x, y, M; c = cgrad(colors), framestyle = :box, kw...)
    savefig(p, path)
    show && display(p)
    return p
end

# Figures of the two-time overlap O(t1, t2): heat maps (linear and log10) with the ridges,
# the ridge t2_max(t1), both time shifts and the diagonal
function plot_overlap_matrix(folder::String, O::Matrix{Float64}, t_axis::Vector{Float64}, ridges;
                             tag::String, log_floor::Float64 = -25.0, size_heatmap = (1000, 900),
                             size_line = (1200, 600), dpi::Int = 150)
    lims = (t_axis[1], t_axis[end])
    for (file, Z, kw) in (("overlap_heatmap.png", O, (;)),
                          ("overlap_log_heatmap.png", log10.(O), (clims = (log_floor, 0.0),)))
        p = heatmap(t_axis, t_axis, Z; c = cgrad(CMAP_SEQUENTIAL), xlabel = "t2 (free motion)",
                    ylabel = "t1 (with the potential)", title = "Overlap O(t1, t2), " * tag,
                    colorbar_title = "O", size = size_heatmap, dpi = dpi, framestyle = :box,
                    aspect_ratio = :equal, xlims = lims, ylims = lims, kw...)
        plot!(p, ridges.t2_max, t_axis; linecolor = :lime, linewidth = 2, label = "t2_max(t1)",
              legend = :topleft)
        plot!(p, t_axis, ridges.t1_max; linecolor = :magenta, linewidth = 2, label = "t1_max(t2)")
        plot!(p, t_axis, t_axis; linecolor = :white, linewidth = 1, linestyle = :dash, label = "t1 = t2")
        savefig(p, joinpath(folder, file))
        display(p)
    end

    line_kw = (linewidth = 2, legend = :topleft, size = size_line, dpi = dpi, framestyle = :box)
    p = plot(t_axis, ridges.t2_max; label = "t2_max(t1)", xlabel = "t1", ylabel = "t2_max",
             title = "Best-matching free time, " * tag, line_kw...)
    plot!(p, t_axis, t_axis; label = "t2 = t1", linestyle = :dash, linecolor = :gray)
    savefig(p, joinpath(folder, "t2_max.png"))

    shift = ridges.t1_max[end] - t_axis[end]
    p = plot(t_axis, t_axis .- ridges.t2_max; label = "t1 − t2_max(t1)", xlabel = "t",
             ylabel = "time shift", title = "Time shift, " * tag, line_kw...)
    plot!(p, t_axis, ridges.t1_max .- t_axis; label = "t1_max(t2) − t2", linewidth = 2)
    hline!(p, [0.0]; linecolor = :gray, linestyle = :dash, label = "")
    hline!(p, [shift]; linecolor = :red, linestyle = :dot, label = @sprintf("final shift = %.3f", shift))
    savefig(p, joinpath(folder, "time_shift.png"))

    p = plot(t_axis, diag(O); xlabel = "t", ylabel = "O(t, t)", title = "Diagonal overlap, " * tag,
             legend = false, size = size_line, dpi = dpi, framestyle = :box)
    savefig(p, joinpath(folder, "overlap_diagonal.png"))
    display(p)
    return nothing
end

# Energy contours H(x, p) = E for phase-space frames, rescaled into the colour range of the heat map
function energy_contours(pot, m, xlims, plims, e0; n::Int = 1000, E_min = 0.2, E_step = 0.2)
    xc = LinRange(xlims[1], xlims[2], n)
    pc = LinRange(plims[1], plims[2], n)
    H  = [p^2 / (2 * m) + pot(x) for x in xc, p in pc]
    levels = collect(range(E_min, plims[2]^2 / (2 * m) + 1.0; step = E_step))
    scale  = 2 * maximum(H)
    return (x = xc, p = pc, H = H ./ scale, levels = levels ./ scale, e0 = e0 / scale)
end

# Phase-space frame: heat map of W[i_x, i_p] with energy contours and the marginals ρx, ρp drawn
# along the axes. Rows i ≥ i_boost of W and ρx are multiplied by boost IN PLACE.
function save_phase_space_frame(path::String, x, p, W::Matrix{Float64}, ρx::Vector{Float64},
                                ρp::Vector{Float64}, contours; boost::Float64 = 1.0,
                                i_boost = nothing, xlims, plims, kw...)
    if i_boost !== nothing && boost != 1.0
        @views W[i_boost:end, :] .*= boost
        @views ρx[i_boost:end]   .*= boost
    end
    hm = heatmap(x, p, transpose(W); c = cgrad([:purple, :blue, :white, :red, :deeppink],
                                               [0.0, 0.375, 0.5, 0.625, 1.0]),
                 xlabel = "x", ylabel = "p", xlims = xlims, ylims = plims, framestyle = :box, kw...)
    contour!(contours.x, contours.p, transpose(contours.H); levels = [contours.e0],
             linecolor = :yellow, linewidth = 2.0, linealpha = 0.6, colorbar_entry = false)
    contour!(contours.x, contours.p, transpose(contours.H); levels = contours.levels,
             linecolor = :gray, linealpha = 0.6, linewidth = 0.8, colorbar = false,
             colorbar_entry = false)
    plot!(x, ρx .+ plims[1]; color = :green, linewidth = 2.0, label = "")
    plot!(ρp .+ xlims[1], p; color = :green, linewidth = 2.0, label = "", legend = false)
    savefig(hm, path)
    return hm
end
