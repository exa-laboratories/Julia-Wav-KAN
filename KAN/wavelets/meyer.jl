module Meyer

export MeyerWavelet

include("../../utils.jl")
include("../wavelets/meyer_utils.jl")

using Flux, CUDA, KernelAbstractions, Tullio
using .MeyerUtils: MeyerAux
using .UTILS: node_mul_1D, node_mul_2D

bool_2D = parse(Bool, get(ENV, "2D", "false"))
node = bool_2D ? node_mul_2D : node_mul_1D

struct MWavelet
    σ
    b
    pi
    norm
    weights
    aux
end

function MeyerWavelet(σ, b, weights, reshape_bool=false)
    normalisation = Float32.([1 / sqrt(σ)])
    bias = Float32.([b])
    return MWavelet(Float32.([σ]), bias, Float32.([π]), normalisation, weights, MeyerAux())
end

function (w::MWavelet)(x)
    ω = abs.((x .- w.b) ./ w.σ)
    sin_term = sin.(ω .* w.pi)
    meyer_term = w.aux(ω)
    y = @tullio out[i, o, b] := sin_term[i, o, b] * meyer_term[i, o, b]
    y = y .* w.norm

    return w.node(y, w.weights)
end

Flux.@functor MWavelet

end