module Meyer

export MeyerWavelet

struct MeyerWavelet
    σ::Float32
    y::Float32
    normalisation::Float32
    weights
end

function MeyerWavelet(σ, y, weights)
    normalisation = 1 / sqrt(σ)
    return MeyerWavelet(σ, y, normalisation, weights)
end

function nu(x)
    return x^4 * (35 - 84 * x + 70 * x^2 - 20 * x^3)
end

function meyer_aux(x)
    if x <= 0.5
        return 1
    elseif x <= 1
        return cos(π * nu(2 * x - 1) / 2)
    else
        return 0
    end
end

function (w::MeyerWavelet)(x)
    ω = abs.((x .- w.y) ./ w.σ)
    
    function scalar_eval(z, weight)
        return sin(π * z) * meyer_aux(z) * weight * w.normalisation
    end

    return scalar_eval.(ω, w.weights)
end

end