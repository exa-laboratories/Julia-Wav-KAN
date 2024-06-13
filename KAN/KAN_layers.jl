module layers

include("./wavelets/mexican_hat.jl")
include("./wavelets/morlet.jl")
include("./wavelets/derivative_of_gaussian.jl")
include("./wavelets/shannon.jl")
include("./wavelets/meyer.jl")

export KANdense

using Flux, CUDA, KernelAbstractions, Tullio
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

# RNO takes 1D inputs, Transformer takes 2D inputs
function scale_translate_1D(x, scale, translation)
    return @tullio out[i, o, b] := (x[i, o, b] - translation[i, o]) / scale[i, o]
end

function scale_translate_2D(x, scale, translation)
    return @tullio out[i, o, l, b] := (x[i, o, l, b] - translation[i, o]) / scale[i, o]
end

function reshape_1D(x, out_size)
    return repeat(reshape(x, size(x, 1), 1, size(x, 2)), 1, out_size, 1)
end

function reshape_2D(x, out_size)
    return repeat(reshape(x, size(x, 1), 1, size(x, 2), size(x, 3)), 1, out_size, 1, 1)
end

reshape_fcn = bool_2D ? reshape_2D : reshape_1D
norm_permute = bool_2D ? x -> reshape(x, size(x, 2), size(x, 1), size(x, 3)) : x -> x
scale_translate_fcn = bool_2D ? scale_translate_2D : scale_translate_1D

struct KANdense_layer
    transform
    output_layer
    batch_norm
    scale
    translation
end

function KANdense(input_size, output_size, wavelet_name, base_activation, batch_norm)
    wavelet_weights = Flux.kaiming_uniform(input_size, output_size)
    wavelet = wavelet_mapping[wavelet_name](wavelet_weights)
    activation = act_mapping[base_activation]
    # output_layer = Flux.Dense(input_size, output_size, activation)
    output_layer = nothing
    batch_norm_layer = batch_norm ? Flux.BatchNorm(output_size, NNlib.relu) : identity

    translation = zeros(input_size, output_size)
    scale = ones(input_size, output_size)

    return KANdense_layer(wavelet, output_layer, batch_norm_layer, scale, translation)
end

function (l::KANdense_layer)(x) 

    x_expanded = reshape_fcn(x, size(l.translation, 2))
    x_expanded = scale_translate_fcn(x_expanded, l.scale, l.translation)

    y = l.transform(x_expanded)
    #z = l.output_layer(x)
    out = y #+ z
    out = l.batch_norm(norm_permute(out))
    return norm_permute(out)
end

Flux.@functor KANdense_layer

end