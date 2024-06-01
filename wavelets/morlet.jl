module Morlet

export MorletWavelet

using Flux, CUDA, KernelAbstractions, Tullio

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
    y = @tullio out[i,b] := real[i,b] * envelope[i,b]
    return @tullio out[o,b] := w.weights[i,o,1] * y[i,b]
end

Flux.@functor MW

end