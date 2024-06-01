module Morlet

export MorletWavelet

using Flux, CUDA, KernelAbstractions, Tullio

struct MW
    γ
    weights
end

function MorletWavelet(γ, weights)
    return MW(γ, weights)
end

function (w::MW)(x)
    real = cos.(w.γ .* x)
    envelope = exp.(-x.^2 ./ 2.0)
    y = @tullio out[i,b] := real[i,b] * envelope[i,b]
    return @tullio out[o,b] := w.weights[i,o,1] * y[i,b]
end

Flux.@functor MW

end