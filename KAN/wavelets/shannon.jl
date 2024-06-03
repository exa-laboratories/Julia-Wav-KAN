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

struct SW
    σ
    b
    sinc_norm
    cos_norm
    norm
    weights
end

function ShannonWavelet(σ, b, weights)
    bias = Float32.([b])
    sinc_norm = Float32.([2.0 * π])
    cos_norm = Float32.([π / 3.0])
    base_norm = Float32.([2.0 / sqrt(σ)])
    return SW(Float32.([σ]), bias, sinc_norm, cos_norm, base_norm, weights)
end


function (w::SW)(x)
    ω = (x .- w.b) ./ w.σ
    first_term = sinc.(ω .* w.sinc_norm)
    second_term = cos.(ω .* w.cos_norm)
    y = batch_mul(first_term, second_term)
    y = y .* w.norm

    return node(y, w.weights)
end

Flux.@functor SW

end