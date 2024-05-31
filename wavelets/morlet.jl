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
    function mor_fcn(z)
        real = cos.(w.γ .* z)
        envelope = exp.(-z .^ 2 ./ 2)
        return real .* envelope
    end
    return w.weights * mor_fcn(x)
end

Flux.@functor MW

end