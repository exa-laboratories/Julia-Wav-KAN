module Meyer

export MeyerWavelet

include("../../utils.jl")
include("../wavelets/meyer_utils.jl")

using Flux, CUDA, KernelAbstractions, Tullio
using .MeyerUtils: MeyerAux
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

pie = [π] |> gpu

struct MWavelet
    weights::AbstractArray
    aux
end

function MeyerWavelet(weights)
    return MWavelet(weights, MeyerAux)
end

function (w::MWavelet)(x)
    ω = abs.(x)
    sin_term = sin.(ω .* pie)
    meyer_term = w.aux(ω)
    y = batch_mul(sin_term, meyer_term)

    return node(y, w.weights)
end

Flux.@functor MWavelet

end