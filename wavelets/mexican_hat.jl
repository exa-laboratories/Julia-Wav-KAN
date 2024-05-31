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
    function MH_fcn(z)
        term1 = 1.0 .- (z.^2 ./ w.σ^2)
        term2 = exp.(-z.^2 ./ (2 * w.σ^2))
        return term1 .* term2 .* w.norm
    end
    return w.weights * MH_fcn(x)
end

Flux.@functor MexicanHatWavelet

end 