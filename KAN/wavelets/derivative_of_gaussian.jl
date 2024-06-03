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

struct DoGWavelet
    σ
    exp_norm
    base_norm
    weights
end

function DoGWavelet(σ, weights)
    exp_norm = Float32.([-1 / (2 * σ^2)])
    normalisation = Float32.([1 / (σ * sqrt(2 * π))])
    return DoGWavelet(σ, exp_norm, normalisation, weights)
end

function (w::DoGWavelet)(x)
    exp_term = exp.(x .* w.exp_norm)
    y = batch_mul(x, exp_term)
    y = y .* w.base_norm
    return node(y, w.weights)
end

Flux.@functor DoGWavelet

end