module KAN_Transform_Layers

export encoder_layers, decoder_layers

include("../../waveletKAN/KAN_layers.jl")

using NNlib: softmax, batched_mul
using Flux
using Flux: Chain, BatchNorm, LayerNorm, Dense, Dropout
using CUDA, KernelAbstractions, Tullio
using .layers: KANdense
using ConfParser

struct mh_attn
    Wq
    Wk
    Wv
    sqrt_D
    query_M
end

function multi_head_attention(wavelet_name, batch_norm)
    d_model = parse(Int, get(ENV, "d_model", "512"))
    base_activation = get(ENV, "activation", "relu")
    nhead = parse(Int, get(ENV, "nhead", "8"))
    d_k = d_model ÷ nhead
    query_mul = [d_k ^ (-0.5)]
    sqrt_d_model = [sqrt(d_model)]

    Wq = KANdense(d_model, d_model, wavelet_name, base_activation, batch_norm)
    Wk = KANdense(d_model, d_model, wavelet_name, base_activation, batch_norm)
    Wv = KANdense(d_model, d_model, wavelet_name, base_activation, batch_norm)   
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
    d_model = parse(Int, get(ENV, "d_model", "512"))
    dim_feedforward = parse(Int, get(ENV, "dim_feedforward", "2048"))
    dropout = parse(Float32, get(ENV, "dropout", "0.1"))
    base_activation = get(ENV, "activation", "relu")

    feed_forward = [
        KANdense(d_model, dim_feedforward, wavelet_name, base_activation, batch_norm),
        Dropout(dropout),
        KANdense(dim_feedforward, d_model, wavelet_name, base_activation, batch_norm),
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
    d_model = parse(Int, get(ENV, "d_model", "512"))
    dim_feedforward = parse(Int, get(ENV, "dim_feedforward", "2048"))
    dropout = parse(Float32, get(ENV, "dropout", "0.1"))
    base_activation = get(ENV, "activation", "relu")

    feed_forward = [
        KANdense(d_model, dim_feedforward, wavelet_name, base_activation, batch_norm),
        Dropout(dropout),
        KANdense(dim_feedforward, d_model, wavelet_name, base_activation, batch_norm),
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