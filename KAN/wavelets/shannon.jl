module Shannon

export ShannonWavelet

include("../../utils.jl")

using Flux, CUDA, KernelAbstractions, Tullio
using .UTILS: node_mul_1D, node_mul_2D

bool_2D = parse(Bool, get(ENV, "2D", "false"))
node = bool_2D ? node_mul_2D : node_mul_1D

function batch_mul_1D(x, y)
    return @tullio out[i, o, b] := x[i, o, b] * y[i, o, b]
end

function batch_mul_2D(x, y)
    return @tullio out[i, o, l, b] := x[i, o, l, b] * y[i, o, l, b]
end

batch_mul = bool_2D ? batch_mul_2D : batch_mul_1D

sinc_norm = [2.0 * π] |> gpu
cos_norm = [π / 3.0] |> gpu
base_norm = [2.0] |> gpu

struct SW
    weights
end

function ShannonWavelet(weights)
    return SW(weights)
end


function (w::SW)(x)
    ω = x
    first_term = sinc.(ω .* sinc_norm)
    second_term = cos.(ω .* cos_norm)
    y = batch_mul(first_term, second_term)
    y = y .* base_norm

    return node(y, w.weights)
end

Flux.@functor SW

end