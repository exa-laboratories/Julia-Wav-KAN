module Morlet

export MorletWavelet

struct MW
    γ::Float32
    weights
end

function MorletWavelet(γ, weights)
    return MW(γ, weights)
end

function (w::MW)(x)
    function scalar_eval(z, weight)
        real = cos(w.γ * z)
        envelope = exp(-z ^ 2 / 2)
        return real * envelope * weight
    end
    return scalar_eval.(x, w.weights)
end

end