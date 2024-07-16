module TransformerModel

export createTransformer

include("./transformer_layers.jl")

using Flux
using Flux: Chain, Dense
using ConfParser
using NNlib: hardtanh
using CUDA, KernelAbstractions, Tullio
using .Transform_Layers: encoder_layers, decoder_layers

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

struct Transformer
    position_encoding
    encoder
    decoder
    output_layer
end

function createTransformer()
    num_encoder_layers = parse(Int, get(ENV, "num_encoder_layers", "6"))
    num_decoder_layers = parse(Int, get(ENV, "num_decoder_layers", "6"))
    d_model = parse(Int, get(ENV, "d_model", "512"))

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

