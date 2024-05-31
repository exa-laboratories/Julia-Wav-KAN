module MexicanHat

export MexicanHatWavelet

struct MexicanHatWavelet
    σ::Float32
    norm
end

function MexicanHatWavelet(σ)
    normalisation = 2 / sqrt((3 * σ * sqrt(π)))
    return MexicanHatWavelet(σ, normalisation)
end

function (w::MexicanHatWavelet)(x)
    return (1 - x^2 / w.σ^2) * exp(-x^2 / (2 * w.σ^2)) * w.norm
end

end 