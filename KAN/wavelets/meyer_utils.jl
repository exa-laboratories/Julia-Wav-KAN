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

eps = [1e-6] |> gpu
half = [0.5] |> gpu
one = [1.0] |> gpu
two = [2.0] |> gpu
zero = [0.0] |> gpu
pie = [π] |> gpu
const_term = [35] |> gpu
lin_term = [-84] |> gpu
quad_term = [70] |> gpu
cubic_term = [-20] |> gpu

function nu(x)
    term1 = const_term .+ (lin_term .* x) .+ (quad_term .* x.^2) .+ (cubic_term .* x.^3)
    term2 = x .^ 4
    return nu_mul(term1, term2)
end


function smooth_step(z, a, b)
    return (one .+ tanh.((z .- a) ./ (b .- a))) .* half
end

function MeyerAux(x)

    transition_0_5 = smooth_step(x, half, half + eps)
    transition_1 = smooth_step(x, one, one + eps)

    term1 = (one .- transition_0_5)  # term for x <= 0.5
    term2_cos = cos.(pie .* nu(two .* x .- one) ./ two)
    term2_one = one .- transition_1
    term2 = three_mul(term2_cos, term2_one, transition_0_5)  # term for 0.5 < x <= 1
    term3 = transition_1 .* zero  # term for x > 1

    return term1 + term2 + term3
end

Flux.@functor MeyerAux

end