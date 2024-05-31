module Meyer

export MeyerWavelet

struct MeyerWavelet
    σ::Float32 # Scale
    y::Float32 # Translation
    normalisation::Float32
    weights
end

function MeyerWavelet(σ, y, weights)
    normalisation = 1 / sqrt(σ)
    return MeyerWavelet(σ, y, normalisation, weights)
end

function sinc(x, eps=1e-6)
    return sin(π * x) / ((π * x) + eps)
end


function (w::MeyerWavelet)(x)
    ω = (x .- w.y) ./ w.σ
    
    function scalar_eval(z, weight)
        return 2 * w.normalisation * sinc(2 * π * z) * cos(π * z / 3) * weight
    end

    return scalar_eval.(ω, w.weights)
end

end