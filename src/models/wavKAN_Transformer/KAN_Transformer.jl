module KANTransformerModel

export create_KAN_Transformer

ENV["2D"] = true # Set to true for 2D matrix muls

include("./Klayers.jl")
include("../../waveletKAN/KAN_layers.jl")

using Flux
using Flux: Chain, Dense
using NNlib: hardtanh
using CUDA, KernelAbstractions, Tullio
using .KAN_Transform_Layers: encoder_layers, decoder_layers
using .layers: KANdense

struct PositionEncoding
    pe_vector
end

function PositionalEncoding()
    d_model = parse(Int, get(ENV, "d_model", "512"))
    max_len = parse(Int, get(ENV, "max_len", "5000"))

    pe_vector = zeros(Float32, d_model, max_len)
    position = range(1, max_len)
    div_term = exp.(-log(10000.0) .* range(1, d_model, step=2) ./ d_model)
    div_term = reshape(div_term, 1, floor(Int, d_model/2))
    pe_vector[1:2:end, :] = transpose(sin.(position .* div_term))
    pe_vector[2:2:end, :] = transpose(cos.(position .* div_term))
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
    base_activation = get(ENV, "activation", "relu")
    num_encoder_layers = parse(Int, get(ENV, "num_encoder_layers", "6"))
    num_decoder_layers = parse(Int, get(ENV, "num_decoder_layers", "6"))
    d_model = parse(Int, get(ENV, "d_model", "512"))

    position_encoding = PositionalEncoding()
    encoder = [encoder_layers(encoder_wavlet_names[i], encoder_batch_norm) for i in 1:num_encoder_layers]
    decoder = [decoder_layers(decoder_wavlet_names[i], decoder_batch_norm) for i in 1:num_decoder_layers]
    output_layer = KANdense(d_model, 1, output_wavelet_name, base_activation, output_batch_norm)

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
    prediction = m.output_layer(tgt)
    prediction = reshape(prediction, size(prediction, 2), size(prediction, 3))

    return prediction
end

Flux.@functor KAN_Transformer

end