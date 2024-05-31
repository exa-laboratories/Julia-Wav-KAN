module MexicanHat

export MexicanHatWavelet

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
    function scalar_eval(z, weight)
        return (1 - (z^2 / w.σ^2)) * exp(-z ^ 2 / (2 * w.σ^2)) * weight * w.norm
    end
    return scalar_eval.(x, w.weights)
end

end 