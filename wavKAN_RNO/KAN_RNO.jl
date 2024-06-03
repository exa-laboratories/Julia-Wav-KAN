module KAN_RecurrentNO

export create_KAN_RNO

include("../KAN/KAN_layers.jl")

using Flux
using Flux: Chain, Dense
using CUDA, KernelAbstractions
using NNlib
using ConfParser
using Tullio
using .layers: KANdense

# Use same baseline architecture
conf = ConfParse("wavKAN_RNO/KAN_RNO_config.ini") 
parse_conf!(conf)

n_hidden = parse(Int, retrieve(conf, "Architecture", "n_hidden"))
num_layers = parse(Int, retrieve(conf, "Architecture", "num_layers")) 
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

struct KAN_RNO
    output_layers
    hidden_layers
    dt
    T::Int64
end

function create_KAN_RNO(input_dim::Int64, output_dim::Int64, input_size::Int64, wavelet_names, batch_norm)
    hidden_units = [n_hidden] * num_layers
    layer_output = [input_dim + output_dim + n_hidden, hidden_units..., output_dim]
    layer_hidden = [n_hidden + output_dim, hidden_units..., n_hidden]

    out_layers_list = [KANdense(layer_output[i], layer_output[i+1], wavelet_names[i], base_activation, batch_norm, arg_mapping[wavelet_names[i]], true) for i in 1:length(layer_output)-1]
    hid_layers_list = [KANdense(layer_hidden[i], layer_hidden[i+1], wavelet_names[i], base_activation, batch_norm, arg_mapping[wavelet_names[i]], true) for i in 1:length(layer_hidden)-1]

    dt = Float32.([1/(input_size-1)])

    return KAN_RNO(out_layers_list, hid_layers_list, dt, input_size)
end

function init_hidden(m::KAN_RNO, batch_size)
    return zeros(Float32, n_hidden, batch_size) |> gpu
end

function fwd_pass(m::KAN_RNO, x, y, hidden)
    x = reshape(x, 1, length(x))
    y = reshape(y, 1, length(y))
    
    # Hidden states
    h0 = init_hidden(m, size(x,2))
    h = vcat(y, hidden)
    for layer in m.hidden_layers
        h = layer(h)
    end
    h = (h .* m.dt) + h0
    
    # Output
    output = vcat(y, (y - x) ./ m.dt, h)
    for layer in m.output_layers
        output = layer(output)
    end

    return output, h
end

function (m::KAN_RNO)(x, y_true)
    
    # Predict from first time step
    y = reshape(y_true[1, :], 1, length(y_true[1, :]))

    hidden = init_hidden(m, size(x)[end])
    for t in 2:m.T
        out, hidden = fwd_pass(m, x[t, :], x[t-1, :], hidden)
        y = vcat(y, out)
    end
    return y
end

Flux.@functor KAN_RNO

end

# Test
using .KAN_RecurrentNO
using Flux, CUDA, KernelAbstractions

wavelet_names = ["MexicanHat", "Morlet"]
batch_norm = true
input_dim = 2

m = create_KAN_RNO(1, 1, 2, wavelet_names, batch_norm) |> gpu

x = rand(Float32, 2, 2) |> gpu
y = rand(Float32, 1, 2) |> gpu

m(x, y)
