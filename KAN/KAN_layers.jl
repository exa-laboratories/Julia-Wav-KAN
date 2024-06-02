module layers

include("./wavelets/mexican_hat.jl")
include("./wavelets/morlet.jl")
include("./wavelets/derivative_of_gaussian.jl")
include("./wavelets/shannon.jl")
include("./wavelets/meyer.jl")

export KANdense

using Flux, CUDA, KernelAbstractions
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
    scale
    translation
end

function KANdense(input_size, output_size, wavelet_name, base_activation, batch_norm, args)
    wavelet_weights = Flux.kaiming_uniform(input_size, output_size)
    wavelet_weights = Float32.(wavelet_weights)
    wavelet = wavelet_mapping[wavelet_name](args..., wavelet_weights)
    activation = act_mapping[base_activation]
    output_layer = Flux.Dense(input_size, output_size, activation)
    batch_norm_layer = batch_norm ? Flux.BatchNorm(output_size, NNlib.relu) : identity

    translation = zeros(input_size, output_size)
    scale = ones(input_size, output_size)

    return KANdense(wavelet, output_layer, batch_norm_layer, scale, translation)
end

function (l::KANdense)(x) 
    println("x: ", size(x))

    x_expanded = reshape(x, size(x, 1), 1, size(x, 2))
    x_expanded = repeat(x_expanded, 1, size(l.translation, 2), 1)
    translation_expanded = repeat(l.translation, 1, 1, size(x, 2))
    scale_expanded = repeat(l.scale, 1, 1, size(x, 2))

    println("x_expanded: ", size(x_expanded))
    println("translation_expanded: ", size(translation_expanded))
    println("scale_expanded: ", size(scale_expanded))

    println("translation_expanded: ", typeof(translation_expanded))
    println("scale_expanded: ", typeof(scale_expanded))
    println("x_expanded: ", typeof(x_expanded))

    x_expanded = (x_expanded - translation_expanded) ./ scale_expanded 

    y = l.transform(x_expanded)
    z = l.output_layer(x)
    out = y + z
    out = l.batch_norm(out)
    return out
end

Flux.@functor KANdense

end