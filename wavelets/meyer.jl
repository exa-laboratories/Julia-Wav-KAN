module Meyer

export MeyerWavelet

using Flux, CUDA, KernelAbstractions, Tullio

struct MeyerWavelet
    σ
    b
    pi
    norm
    weights
end

function MeyerWavelet(σ, b, weights)
    normalisation = Float32.([1 / sqrt(σ)])
    bias = Float32.([b])
    pi = Float32.([π])
    return MeyerWavelet(σ, bias, pi, normalisation, weights)
end

function nu(x)
    return x^4 * (35 - 84 * x + 70 * x^2 - 20 * x^3)
end

function meyer_aux(x, eps=1e-6)
    function smooth_step(z, a, b)
        return 0.5 * (1 + tanh((z - a) / (b - a)))
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
    sin_term = sin.(ω .* w.pi)
    meyer_term = meyer_aux.(ω)
    y = @tullio out[i,b] := sin_term[i,b] * meyer_term[i,b]
    y = y .* w.norm

    return @tullio out[o,b] := w.weights[i,o,1] * y[i,b]
end

Flux.@functor MeyerWavelet

end