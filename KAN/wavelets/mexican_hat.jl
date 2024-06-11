module MexicanHat

export MexicanHatWavelet

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

struct MHWavelet
    σ
    one
    exp_norm
    norm
    weights
end

function MexicanHatWavelet(σ, weights)
    exp_norm = [-1 / (2 * σ^2)]
    normalisation = [2 / sqrt((3 * σ * sqrt(π)))]
    return MHWavelet([σ], [1], exp_norm, normalisation, weights)
end

function (w::MHWavelet)(x)
    term_1 = w.one .- (x.^2 ./ (w.σ.^2))
    term_2 = exp.(x.^2 .* w.exp_norm)
    y = batch_mul(term_1, term_2)
    y = y .* w.norm
    return node(y, w.weights)
end

Flux.@functor MHWavelet

end 