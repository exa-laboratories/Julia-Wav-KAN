module DoG

export DoGWavelet

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
    function scalar_eval(z, weight)
        return - z * exp(-z^2 / (2 * w.σ^2)) * weight * w.norm
    end
    return scalar_eval.(x, w.weights)
end

end