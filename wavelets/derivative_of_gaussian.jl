module DoG

export DoGWavelet

using Flux

struct DoGWavelet
    σ::Float32
    norm::Float32
    weights
end

function DoGWavelet(σ, weights)
    normalisation = 1 / (σ * sqrt(2 * π))
    return DoGWavelet(σ, normalisation, weights)
end

function (w::DoGWavelet)(x)
    function DoG_fcn(z)
        return - z .* exp.(-z.^2 / (2 * w.σ^2)) .* w.norm
    end
    return w.weights * DoG_fcn(x)
end

Flux.@functor DoGWavelet

end