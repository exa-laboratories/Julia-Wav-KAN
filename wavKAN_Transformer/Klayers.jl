module KAN_Transform_Layers

export encoder_layers, decoder_layers

include("../KAN/KAN_layers.jl")

using NNlib: softmax, batched_mul
using Flux
using Flux: Chain, BatchNorm, LayerNorm, Dense, Dropout
using ConfParser
using CUDA, KernelAbstractions, Tullio
using .layers: KANdense


conf = ConfParse("wavKAN_Transformer/KAN_Transformer_config.ini")
parse_conf!(conf)

d_model = parse(Int, retrieve(conf, "Architecture", "d_model"))
nhead = parse(Int, retrieve(conf, "Architecture", "nhead"))
dim_feedforward = parse(Int, retrieve(conf, "Architecture", "dim_feedforward"))
max_len = parse(Int, retrieve(conf, "Architecture", "max_len"))
dropout = parse(Float32, retrieve(conf, "Architecture", "dropout"))
base_activation = retrieve(conf, "Architecture", "activation")

d_k = d_model ÷ nhead
query_mul = Float32.([d_k ^ (-0.5)]) 
sqrt_d_model = Float32.([sqrt(d_model)]) 

# Activation mapping
wavelet_conf = ConfParse("wavelet_config.ini")
parse_conf!(wavelet_conf)

arg_mapping = Dict(
    "MexicanHat" => parse(Float32, retrieve(wavelet_conf, "MexicanHat", "sigma")),
    "Morlet" => parse(Float32, retrieve(wavelet_conf, "Morlet", "gamma")),
    "DerivativeOfGaussian" => parse(Float32, retrieve(wavelet_conf, "DerivativeOfGaussian", "sigma")),
    "Shannon" => (parse(Float32, retrieve(wavelet_conf, "Shannon", "sigma")), parse(Float32, retrieve(wavelet_conf, "Shannon", "bias"))),
    "Meyer" => (parse(Float32, retrieve(wavelet_conf, "Meyer", "sigma")), parse(Float32, retrieve(wavelet_conf, "Meyer", "bias")))
)

struct mh_attn
    Wq
    Wk
    Wv
    sqrt_D
    query_M
end

function multi_head_attention(wavelet_name, batch_norm)
    Wq = KANdense(d_model, d_model, wavelet_name, base_activation, batch_norm, arg_mapping[wavelet_name])
    Wk = KANdense(d_model, d_model, wavelet_name, base_activation, batch_norm, arg_mapping[wavelet_name])
    Wv = KANdense(d_model, d_model, wavelet_name, base_activation, batch_norm, arg_mapping[wavelet_name])   
    return mh_attn(Wq, Wk, Wv, sqrt_d_model, query_mul)
end

function scaled_dot_product_attention(query, key, value, sd)
    scores = @tullio k[i, j, b] := query[i, j, b] * key[i, t, b]
    scores = scores ./ sd
    p_attn = softmax(scores, dims=1)
    return @tullio out[i, j, b] := p_attn[i, j, b] * value[i, t, b]
end

function (att::mh_attn)(x, y, z)
    # Print device
    query = att.Wq(x)
    key = att.Wk(y)
    value = att.Wv(z)
    query = query .* att.query_M
    out = scaled_dot_product_attention(query, key, value, att.sqrt_D)
    return out
end

Flux.@functor mh_attn

struct encoder_layer
    self_attn
    feed_forward
    norm1
    norm2
end

function encoder_layers(wavelet_name, batch_norm)
    feed_forward = [
        KANdense(d_model, dim_feedforward, wavelet_name, base_activation, batch_norm, arg_mapping[wavelet_name]),
        Dropout(dropout),
        KANdense(dim_feedforward, d_model, wavelet_name, base_activation, batch_norm, arg_mapping[wavelet_name]),
        Dropout(dropout)
    ]
    norm1 = LayerNorm(d_model)
    norm2 = LayerNorm(d_model)
    return encoder_layer(multi_head_attention(wavelet_name, batch_norm), feed_forward, norm1, norm2)
end

function (l::encoder_layer)(x)
    x = l.norm1(x + l.self_attn(x, x, x))
    z = copy(x)
    for layer in l.feed_forward
        z = layer(z)
    end
    return l.norm2(x + z)
end

struct decoder_layer
    mh_attn
    feed_forward
    norm1
    norm2
    norm3
end

function decoder_layers(wavelet_name, batch_norm)
    feed_forward = [
        KANdense(d_model, dim_feedforward, wavelet_name, base_activation, batch_norm, arg_mapping[wavelet_name]),
        Dropout(dropout),
        KANdense(dim_feedforward, d_model, wavelet_name, base_activation, batch_norm, arg_mapping[wavelet_name]),
        Dropout(dropout)
    ]
    norm1 = LayerNorm(d_model)
    norm2 = LayerNorm(d_model)
    norm3 = LayerNorm(d_model)
    return decoder_layer(multi_head_attention(wavelet_name, batch_norm), feed_forward, norm1, norm2, norm3)
end

function (l::decoder_layer)(x, memory)
    x = l.norm1(x + l.mh_attn(x, x, x))
    x = l.norm2(x + l.mh_attn(x, memory, memory))
    z = copy(x)
    for layer in l.feed_forward
        z = layer(z)
    end
    return l.norm3(x + z)
end

Flux.@layer encoder_layer
Flux.@layer decoder_layer

end