module MeyerUtils

export MeyerAux

using Flux, CUDA, KernelAbstractions, Tullio

struct nu
    const_term
    lin_term
    quad_term
    cubic_term
end

function nu()
    const_term = Float32.([35])
    lin_term = Float32.([-84])
    quad_term = Float32.([70])
    cubic_term = Float32.([-20])
    return nu(const_term, lin_term, quad_term, cubic_term)
end

function (n::nu)(x)
    term1 = n.const_term .+ (n.lin_term .* x) .+ (n.quad_term .* x.^2) .+ (n.cubic_term .* x.^3)
    term2 = x .^ 4
    return @tullio out[i,b] := term1[i,b] * term2[i,b]
end

struct smooth_step
    half
    one
end

function smooth_step()
    half = Float32.([0.5])
    one = Float32.([1.0])
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
    eps = Float32.([1e-6])
    half = Float32.([0.5])
    one = Float32.([1.0])
    two = Float32.([2.0])
    zero = Float32.([0.0])
    return MeyerAux(nu(), smooth_step(), eps, half, one, two, zero, Float32.([π]))
end

function (m::MeyerAux)(x)

    transition_0_5 = m.smooth_step(x, m.half, m.half + m.eps)
    transition_1 = m.smooth_step(x, m.one, m.one + m.eps)

    term1 = (m.one .- transition_0_5)  # term for x <= 0.5
    term2_cos = cos.(m.pi .* m.nu(m.two .* x .- m.one) ./ m.two)
    term2_one = m.one .- transition_1
    term2 = @tullio out[i,b] := term2_cos[i,b] * term2_one[i,b] * transition_0_5[i,b]  # term for 0.5 < x <= 1
    term3 = transition_1 .* m.zero  # term for x > 1

    return term1 + term2 + term3
end

Flux.@functor nu
Flux.@functor smooth_step
Flux.@functor MeyerAux

end