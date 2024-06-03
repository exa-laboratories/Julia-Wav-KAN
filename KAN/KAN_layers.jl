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

bool_2D = parse(Bool, get(ENV, "2D", "false"))

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

struct KANdense_layer
    transform
    output_layer
    batch_norm
    scale
    translation
    reshape_fcn
    norm_permute
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

    # RNO takes 1D input, else transformer uses 2D input
    reshape_fcn = bool_2D ? x -> repeat(reshape(x, size(x, 1), 1, size(x, 2), size(x, 3)), 1, size(translation, 2), 1, 1) : x -> repeat(reshape(x, size(x, 1), 1, size(x, 2)), 1, size(translation, 2), 1)
    norm_permute = bool_2D ? x -> reshape(x, size(x, 2), size(x, 1), size(x, 3)) : x -> x

    return KANdense_layer(wavelet, output_layer, batch_norm_layer, scale, translation, reshape_fcn, norm_permute)
end

function (l::KANdense_layer)(x) 
    println("x: ", size(x))

    x_expanded = l.reshape_fcn(x)
    translation_expanded = repeat(l.translation, 1, 1, size(x_expanded)[3:end]...)
    scale_expanded = repeat(l.scale, 1, 1, size(x_expanded)[3:end]...)

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
    out = l.batch_norm(l.norm_permute(out))
    return l.norm_permute(out)
end

Flux.@functor KANdense_layer

end