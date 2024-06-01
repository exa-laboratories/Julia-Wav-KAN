module Meyer

export MeyerWavelet

using Flux

struct MeyerWavelet
    σ::Float32
    b::Float32
    normalisation::Float32
    weights
end

function MeyerWavelet(σ, b, weights)
    normalisation = 1 / sqrt(σ)
    return MeyerWavelet(σ, b, normalisation, weights)
end

function nu(x)
    return x^4 * (35 - 84 * x + 70 * x^2 - 20 * x^3)
end

function meyer_aux(x, eps=1e-6)
    function smooth_step(x, a, b)
        return 0.5 * (1 + tanh((x - a) / (b - a)))
    end

    transition_0_5 = smooth_step(x, 0.5, 0.5 + eps)
    transition_1 = smooth_step(x, 1, 1 + eps)

    term1 = (1 - transition_0_5)  # term for x <= 0.5
    term2 = transition_0_5 * (1 - transition_1) * cos(π * nu(2 * x - 1) / 2)  # term for 0.5 < x <= 1
    term3 = transition_1 * 0  # term for x > 1

    return term1 + term2 + term3
end

function (w::MeyerWavelet)(x)
    ω = abs.((x .- w.b) ./ w.σ)
    
    function scalar_eval(z)
        return sin.(π .* z) .* meyer_aux.(z) .* w.normalisation
    end

    return w.weights * scalar_eval(ω) 
end

Flux.@functor MeyerWavelet

end