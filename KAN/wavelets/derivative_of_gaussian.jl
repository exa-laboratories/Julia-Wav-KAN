module DoG

export DoGWavelet

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

exp_norm = [-1 / 2] |> gpu
normalisation = [1 / (sqrt(2 * π))] |> gpu

struct dogWavelet
    weights
end

function DoGWavelet(weights)
    return dogWavelet(weights)
end

function (w::dogWavelet)(x)
    exp_term = exp.(x .* exp_norm)
    y = batch_mul(x, exp_term)
    y = y .* normalisation
    return node(y, w.weights)
end

Flux.@layer dogWavelet

end