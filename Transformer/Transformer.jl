module TransformerModel

export createTransformer, inference

include("./transformer_layers.jl")

using Flux
using Flux: Chain, Dense
using ConfParser
using NNlib: hardtanh
using CUDA, KernelAbstractions, Tullio
using .Transform_Layers: encoder_layers, decoder_layers

conf = ConfParse("Transformer_config.ini")
parse_conf!(conf)

d_model = parse(Int, retrieve(conf, "Architecture", "d_model"))
num_encoder_layers = parse(Int, retrieve(conf, "Architecture", "num_encoder_layers"))
num_decoder_layers = parse(Int, retrieve(conf, "Architecture", "num_decoder_layers"))
max_len = parse(Int, retrieve(conf, "Architecture", "max_len"))
dropout = parse(Float32, retrieve(conf, "Architecture", "dropout"))

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

struct Transformer
    position_encoding
    encoder
    decoder
    output_layer
end

function createTransformer()
    position_encoding = PositionalEncoding()
    encoder = [encoder_layers() for _ in 1:num_encoder_layers]
    decoder = [decoder_layers() for _ in 1:num_decoder_layers]
    output_layer = Dense(d_model, 1)

    return Transformer(position_encoding, Chain(encoder...), decoder, output_layer)
end


function (m::Transformer)(src, tgt)
    
    src = m.position_encoding(src)
    tgt = m.position_encoding(tgt)
    memory = m.encoder(src)
    for layer in m.decoder
        tgt = layer(tgt, memory)
    end
    prediction = m.output_layer(tgt)
    prediction = reshape(prediction, size(prediction, 2), size(prediction, 3))

    return (hardtanh(prediction) .* 0.5) .+ 0.5

end

Flux.@functor Transformer

end

