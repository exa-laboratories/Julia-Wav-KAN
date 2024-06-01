module DoG

export DoGWavelet

using Flux, CUDA, KernelAbstractions, Tullio

struct DoGWavelet
    σ
    exp_norm
    base_norm
    weights
end

function DoGWavelet(σ, weights)
    exp_norm = Float32.([-1 / (2 * σ^2)])
    normalisation = Float32.([1 / (σ * sqrt(2 * π))])
    return DoGWavelet(σ, exp_norm, normalisation, weights)
end

function (w::DoGWavelet)(x)
    exp_term = exp.(x .* w.exp_norm)
    y = @tullio out[i,b] := x[i,b] * exp_term[i,b]
    y = y .* w.base_norm
    return @tullio out[o,b] := w.weights[i,o,1] * y[i,b]  
end

Flux.@functor DoGWavelet

end