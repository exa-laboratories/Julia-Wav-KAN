module Shannon

export ShannonWavelet

using Flux

struct ShannonWavelet
    σ::Float32 # Scale
    b::Float32 # Translation
    normalisation::Float32
    weights
end

function ShannonWavelet(σ, y, weights)
    normalisation = 1 / sqrt(σ)
    return ShannonWavelet(σ, y, normalisation, weights)
end

function sinc(x, eps=1e-6)
    return sin(π * x) / ((π * x) + eps)
end


function (w::ShannonWavelet)(x)
    ω = (x .- w.b) ./ w.σ
    
    function Shann_fcn(z)
        return 2 .* w.normalisation .* sinc.(2 .* π .* z) .* cos.(π .* z ./ 3)
    end

    return  w.weights * Shann_fcn(ω)
end

Flux.@functor ShannonWavelet

end