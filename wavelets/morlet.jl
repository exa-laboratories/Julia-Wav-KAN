module Morlet

export MorletWavelet

using Flux

struct MW
    γ::Float32
    weights
end

function MorletWavelet(γ, weights)
    return MW(γ, weights)
end

function (w::MW)(x)
    function scalar_eval(z)
        real = cos(w.γ * z)
        envelope = exp(-z ^ 2 / 2)
        return real * envelope * weight
    end
    return w.weights * scalar_eval.(x)
end

Flux.@functor MW

end