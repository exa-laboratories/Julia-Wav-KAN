module KANTransformerModel

export create_KAN_Transformer

ENV["2D"] = true # Set to true for 2D matrix muls

include("./Klayers.jl")
include("../KAN/KAN_layers.jl")

using Flux
using Flux: Chain, Dense
using ConfParser
using NNlib: hardtanh
using CUDA, KernelAbstractions, Tullio
using .KAN_Transform_Layers: encoder_layers, decoder_layers
using .layers: KANdense

conf = ConfParse("wavKAN_Transformer/KAN_Transformer_config.ini")
parse_conf!(conf)

d_model = parse(Int, retrieve(conf, "Architecture", "d_model"))
num_encoder_layers = parse(Int, retrieve(conf, "Architecture", "num_encoder_layers"))
num_decoder_layers = parse(Int, retrieve(conf, "Architecture", "num_decoder_layers"))
max_len = parse(Int, retrieve(conf, "Architecture", "max_len"))
dropout = parse(Float32, retrieve(conf, "Architecture", "dropout"))
base_activation = retrieve(conf, "Architecture", "activation")

wavelet_conf = ConfParse("wavelet_config.ini")
parse_conf!(wavelet_conf)

arg_mapping = Dict(
    "MexicanHat" => parse(Float32, retrieve(wavelet_conf, "MexicanHat", "sigma")),
    "Morlet" => parse(Float32, retrieve(wavelet_conf, "Morlet", "gamma")),
    "DerivativeOfGaussian" => parse(Float32, retrieve(wavelet_conf, "DerivativeOfGaussian", "sigma")),
    "Shannon" => (parse(Float32, retrieve(wavelet_conf, "Shannon", "sigma")), parse(Float32, retrieve(wavelet_conf, "Shannon", "bias"))),
    "Meyer" => (parse(Float32, retrieve(wavelet_conf, "Meyer", "sigma")), parse(Float32, retrieve(wavelet_conf, "Meyer", "bias")))
)

struct PositionEncoding
    pe_vector
end

function PositionalEncoding()
    pe_vector = zeros(Float32, d_model, max_len)
    position = Float32.(range(1, max_len))
    div_term = exp.(Float32.(-log(10000.0) .* range(1, d_model, step=2) ./ d_model))
    div_term = reshape(div_term, 1, floor(Int, d_model/2))
    pe_vector[1:2:end, :] = transpose(sin.(position .* div_term))
    pe_vector[2:2:end, :] = transpose(cos.(position .* div_term))
    pe_vector = Float32.(pe_vector) 
    return PositionEncoding(pe_vector)
end

function (pe::PositionEncoding)(x)
    x = reshape(x, 1, size(x, 1), size(x, 2))
    encoding = repeat(pe.pe_vector[:, 1:size(x, 2)], 1, 1, size(x, 3)) 
    return x .+ encoding
end

Flux.@functor PositionEncoding

struct KAN_Transformer
    position_encoding
    encoder
    decoder
    output_layer
end

function create_KAN_Transformer(encoder_wavlet_names, decoder_wavlet_names, encoder_batch_norm, decoder_batch_norm, output_wavelet_name, output_batch_norm)
    position_encoding = PositionalEncoding()
    encoder = [encoder_layers(encoder_wavlet_names[i], encoder_batch_norm[i]) for i in 1:num_encoder_layers]
    decoder = [decoder_layers(decoder_wavlet_names[i], decoder_batch_norm[i]) for i in 1:num_decoder_layers]
    output_layer = KANdense(d_model, 1, output_wavelet_name, base_activation, output_batch_norm, arg_mapping[output_wavelet_name])

    return KAN_Transformer(position_encoding, encoder, decoder, output_layer)
end


function (m::KAN_Transformer)(src, tgt)
    src = m.position_encoding(src)
    tgt = m.position_encoding(tgt)
    for layer in m.encoder
        src = layer(src)
    end
    for layer in m.decoder
        tgt = layer(tgt, src)
    end
    println("end",size(tgt))
    prediction = m.output_layer(tgt)
    prediction = reshape(prediction, size(prediction, 2), size(prediction, 3))

    return (hardtanh(prediction) .* 0.5) .+ 0.5

end

Flux.@functor KAN_Transformer

end

# Test
using .KANTransformerModel
using CUDA, Flux

# 8 encoder, 1 decoder layers
encoder_wavlet_names = ["MexicanHat", "Morlet", "DerivativeOfGaussian", "Shannon", "MexicanHat", "Morlet", "DerivativeOfGaussian", "Shannon"]
decoder_wavlet_names = ["DerivativeOfGaussian"]
encoder_batch_norm = [true, true, true, true, true, true, true, true]
decoder_batch_norm = [true]
output_wavelet_name = "MexicanHat"
output_batch_norm = true

model = create_KAN_Transformer(encoder_wavlet_names, decoder_wavlet_names, encoder_batch_norm, decoder_batch_norm, output_wavelet_name, output_batch_norm) |> gpu
src = rand(Float32, 2, 4) |> gpu
tgt = rand(Float32, 2, 4) |> gpu

loss, grad = Flux.withgradient(m -> sum(m(src, tgt)), model)

println("Loss: ", loss)
