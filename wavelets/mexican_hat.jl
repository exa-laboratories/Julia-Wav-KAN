module MexicanHat

export MexicanHatWavelet

using Flux

struct MexicanHatWavelet
    σ::Float32
    norm::Float32
    weights
end

function MexicanHatWavelet(σ, weights)
    normalisation = 2 / sqrt((3 * σ * sqrt(π)))
    return MexicanHatWavelet(σ, normalisation, weights)
end

function (w::MexicanHatWavelet)(x)
    function scalar_eval(z)
        return (1 - (z^2 / w.σ^2)) * exp(-z ^ 2 / (2 * w.σ^2)) * w.norm
    end
    return w.weights * scalar_eval.(x)
end

Flux.@functor MexicanHatWavelet

end 