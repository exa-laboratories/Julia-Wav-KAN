module MeyerUtils

export MeyerAux

using Flux, CUDA, KernelAbstractions, Tullio

bool_2D = parse(Bool, get(ENV, "2D", "false"))

function nu_mul_1D(x, y)
    return @tullio out[i, o, b] := x[i, o, b] * y[i, o, b]
end

function nu_mul_2D(x, y)
    return @tullio out[i, o, l, b] := x[i, o, l, b] * y[i, o, l, b]
end

function three_mul_1D(x, y, z)
    return @tullio out[i, o, b] := x[i, o, b] * y[i, o, b] * z[i, o, b]
end

function three_mul_2D(x, y, z)
    return @tullio out[i, o, l, b] := x[i, o, l, b] * y[i, o, l, b] * z[i, o, l, b]
end

nu_mul = bool_2D ? nu_mul_2D : nu_mul_1D
three_mul = bool_2D ? three_mul_2D : three_mul_1D

struct nu
    const_term
    lin_term
    quad_term
    cubic_term
end

function nu()
    const_term = [35]
    lin_term = [-84]
    quad_term = [70]
    cubic_term = [-20]
    return nu(const_term, lin_term, quad_term, cubic_term)
end

function (n::nu)(x)
    term1 = n.const_term .+ (n.lin_term .* x) .+ (n.quad_term .* x.^2) .+ (n.cubic_term .* x.^3)
    term2 = x .^ 4
    return nu_mul(term1, term2)
end

struct smooth_step
    half
    one
end

function smooth_step()
    half = [0.5]
    one = [1.0]
    return smooth_step(half, one)
end

function (s::smooth_step)(z, a, b)
    return (s.one .+ tanh.((z .- a) ./ (b .- a))) .* s.half
end

struct MeyerAux 
    nu
    smooth_step
    eps
    half
    one
    two
    zero
    pi
end

function MeyerAux()
    eps = [1e-6]
    half = [0.5]
    one = [1.0]
    two = [2.0]
    zero = [0.0]
    return MeyerAux(nu(), smooth_step(), eps, half, one, two, zero, Float32.([π]))
end

function (m::MeyerAux)(x)

    transition_0_5 = m.smooth_step(x, m.half, m.half + m.eps)
    transition_1 = m.smooth_step(x, m.one, m.one + m.eps)

    term1 = (m.one .- transition_0_5)  # term for x <= 0.5
    term2_cos = cos.(m.pi .* m.nu(m.two .* x .- m.one) ./ m.two)
    term2_one = m.one .- transition_1
    term2 = three_mul(term2_cos, term2_one, transition_0_5)  # term for 0.5 < x <= 1
    term3 = transition_1 .* m.zero  # term for x > 1

    return term1 + term2 + term3
end

Flux.@functor nu
Flux.@functor smooth_step
Flux.@functor MeyerAux

end