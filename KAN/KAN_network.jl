module KAN_models

export KAN

include("./KAN_layers.jl")

using Flux
using .layers: KANdense
using ConfParser
using CUDA, KernelAbstractions

conf = ConfParse("wavelet_config.ini")
parse_conf!(conf)

arg_mapping = Dict(
    "MexicanHat" => parse(Float32, retrieve(conf, "MexicanHat", "sigma")),
    "Morlet" => parse(Float32, retrieve(conf, "Morlet", "gamma")),
    "DerivativeOfGaussian" => parse(Float32, retrieve(conf, "DerivativeOfGaussian", "sigma")),
    "Shannon" => (parse(Float32, retrieve(conf, "Shannon", "sigma")), parse(Float32, retrieve(conf, "Shannon", "bias"))),
    "Meyer" => (parse(Float32, retrieve(conf, "Meyer", "sigma")), parse(Float32, retrieve(conf, "Meyer", "bias")))
)

struct KAN
    layers
end

function KAN(input_size, output_size, hidden_dims, wavelet_names, base_activations, batch_norms)
    layers = []
    for i in 1:length(hidden_dims)
        name = wavelet_names[i]
        layer = KANdense(input_size, hidden_dims[i], name, base_activations[i], batch_norms[i], arg_mapping[name])
        push!(layers, layer)
        input_size = hidden_dims[i]
    end
    name = wavelet_names[end]
    output_layer = KANdense(input_size, output_size, name, base_activations[end], batch_norms[end], arg_mapping[name])
    push!(layers, output_layer)
    return KAN(layers)
end

function (m::KAN)(x)
    for layer in m.layers
        x = layer(x)
    end
    return x
end

Flux.@functor KAN

end
