module layers

include("../wavelets/mexican_hat.jl")
include("../wavelets/morlet.jl")
include("../wavelets/derivative_of_gaussian.jl")
include("../wavelets/shannon.jl")
include("../wavelets/meyer.jl")

export KANdense

using Flux
using Flux: Dense, BatchNorm
using NNlib
using .MexicanHat: MexicanHatWavelet
using .Morlet: MorletWavelet
using .DoG: DoGWavelet
using .Shannon: ShannonWavelet
using .Meyer: MeyerWavelet

act_mapping = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
)

wavelet_mapping = Dict(
    "MexicanHat" => MexicanHatWavelet,
    "Morlet" => MorletWavelet,
    "DerivativeOfGaussian" => DoGWavelet,
    "Shannon" => ShannonWavelet,
    "Meyer" => MeyerWavelet
)

struct KANdense
    transform
    output_layer
    batch_norm
end

function KANdense(input_size, output_size, wavelet_name, base_activation, batch_norm, args)
    wavelet_weights = Flux.kaiming_uniform(output_size, input_size)
    wavelet = wavelet_mapping[wavelet_name](args..., wavelet_weights)
    activation = act_mapping[base_activation]
    output_layer = Flux.Dense(input_size, output_size, activation)
    batch_norm_layer = batch_norm ? Flux.BatchNorm(output_size) : identity
    return KANdense(wavelet, output_layer, batch_norm_layer)
end

function (l::KANdense)(x)
    y = l.transform(x)
    z = l.output_layer(x)
    out = y + z
    out = l.batch_norm(out)
    return out
end

Flux.@functor KANdense

end

# Test the module
using .layers

input_size = 10
output_size = 5
wavelet_name = "MexicanHat"
base_activation = "relu"
batch_norm = false
args = 0.5

layer = KANdense(input_size, output_size, wavelet_name, base_activation, batch_norm, args)
x = rand(input_size)
y = layer(x)
println("Output size: ", length(y))
println("Output: ", y)