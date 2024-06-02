module RecurrentNO

export createRNO

using Flux
using Flux: Chain, Dense
using CUDA, KernelAbstractions
using NNlib
using ConfParser
using Tullio

conf = ConfParse("./RNO_config.ini")
parse_conf!(conf)

n_hidden = parse(Int, retrieve(conf, "Architecture", "n_hidden"))
num_layers = parse(Int, retrieve(conf, "Architecture", "num_layers"))
activation = retrieve(conf, "Architecture", "activation")

# Activation mapping
act_fcn = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
)[activation]

struct RNO
    output_layers
    hidden_layers
    dt
    T::Int64
end

function createRNO(input_dim::Int64, output_dim::Int64, input_size::Int64)
    phi = act_fcn

    hidden_units = [n_hidden] * num_layers
    layer_output = [input_dim + output_dim + n_hidden, hidden_units..., output_dim]
    layer_hidden = [n_hidden + output_dim, hidden_units..., n_hidden]

    out_layers_list = [Dense(layer_output[i], layer_output[i+1], phi) for i in 1:length(layer_output)-2]
    hid_layers_list = [Dense(layer_hidden[i], layer_hidden[i+1], phi) for i in 1:length(layer_hidden)-2]

    out_layers = Chain(out_layers_list..., Dense(layer_output[end-1], layer_output[end]))
    hid_layers = Chain(hid_layers_list..., Dense(layer_hidden[end-1], layer_hidden[end]))

    dt = Float32.([1/(input_size-1)])

    return RNO(out_layers, hid_layers, dt, input_size)
end

function init_hidden(m::RNO, batch_size)
    return zeros(Float32, n_hidden, batch_size) |> gpu
end

function fwd_pass(m::RNO, x, y, hidden)
    x = reshape(x, 1, length(x))
    y = reshape(y, 1, length(y))
    
    # Hidden states
    h0 = init_hidden(m, size(x,2))
    h = vcat(y, hidden)
    h = m.hidden_layers(h) 
    h = (h .* m.dt) + h0
    
    # Output
    output = vcat(y, (y - x) ./ m.dt, h)
    output = m.output_layers(output)

    return output, h
end

function (m::RNO)(x, y_true)
    
    # Predict from first time step
    y = reshape(y_true[1, :], 1, length(y_true[1, :]))

    hidden = init_hidden(m, size(x)[end])
    for t in 2:m.T
        out, hidden = fwd_pass(m, x[t, :], x[t-1, :], hidden)
        y = vcat(y, out)
    end
    return y
end

Flux.@functor RNO

end