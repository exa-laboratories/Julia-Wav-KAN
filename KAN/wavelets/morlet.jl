module Morlet

export MorletWavelet

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

struct MW
    γ
    norm
    weights
end

function MorletWavelet(γ, weights)
    half = Float32.([0.5])
    return MW(Float32.([γ]), half, weights)
end

function (w::MW)(x)
    real = cos.(w.γ .* x)
    envelope = exp.(-x.^2 .* w.norm)
    y = batch_mul(real, envelope)
    return node(y, w.weights)
end

Flux.@functor MW

end