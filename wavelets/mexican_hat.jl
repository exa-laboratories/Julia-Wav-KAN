module MexicanHat

export MexicanHatWavelet

using Flux, CUDA, KernelAbstractions, Tullio

struct MexicanHatWavelet
    σ
    one
    exp_norm
    norm
    weights
end

function MexicanHatWavelet(σ, weights)
    exp_norm = Float32.([-1 / (2 * σ^2)])
    normalisation = Float32.([2 / sqrt((3 * σ * sqrt(π)))])
    return MexicanHatWavelet(Float32.([σ]), Float32.([1]), exp_norm, normalisation, weights)
end

function (w::MexicanHatWavelet)(x)
    term_1 = w.one .- (x.^2 ./ (w.σ.^2))
    term_2 = exp.(x.^2 .* w.exp_norm)
    y = @tullio out[i,b] := term_1[i,b] * term_2[i,b]
    y = y .* w.norm
    return @tullio out[o,b] := w.weights[i,o,1] * y[i,b]
end

Flux.@functor MexicanHatWavelet

end 