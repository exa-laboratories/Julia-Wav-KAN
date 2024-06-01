module Shannon

export ShannonWavelet

using Flux, CUDA, KernelAbstractions, Tullio

struct SW
    σ
    b
    weights
    sinc_norm
    cos_norm
    norm
end

function ShannonWavelet(σ, b, weights)
    
    bias = Float32.([b])
    sinc_norm = Float32.([2.0 * π])
    cos_norm = Float32.([π / 3.0])
    base_norm = Float32.([2.0 / sqrt(σ)])
    return SW(σ, bias, weights, sinc_norm, cos_norm, base_norm)
end


function (w::SW)(x)
    ω = (x .- w.b) ./ w.σ
    first_term = sinc.(ω .* w.sinc_norm)
    second_term = cos.(ω .* w.cos_norm)
    y = @tullio out[i,b] := (first_term[i,b] * second_term[i,b]) 
    y = y .* w.norm

    return @tullio out[o,b] := w.weights[i,o,1] * y[i,b]  
end

Flux.@functor SW

end