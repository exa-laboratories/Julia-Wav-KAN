include("hp_parsing.jl")
include("pipeline/data_processing/data_loader.jl")
include("utils.jl")
include("pipeline/train.jl")
include("Vanilla_RNO/RNO.jl")
include("Vanilla_Transformer/Transformer.jl")
include("wavKAN_RNO/KAN_RNO.jl")
include("wavKAN_Transformer/KAN_Transformer.jl")

using Random
using Flux, CUDA, KernelAbstractions
using Optimisers
using ProgressBars
using BSON: @save
using .training: train_step
using .loaders: get_visco_loader
using .UTILS: loss_fcn, BIC, log_csv
using .RecurrentNO: createRNO
using .TransformerModel: createTransformer
using .KAN_RecurrentNO: create_KAN_RNO
using .KANTransformerModel: create_KAN_Transformer
using .hyperparams: set_hyperparams

NUM_REPETITIONS = 5

model_name = "RNO"
hyperparams = set_hyperparams(model_name)
batch_size = get(ENV, "batch_size", 32)
learning_rate = get(ENV, "learning_rate", 1e-3)
num_epochs = get(ENV, "num_epochs", 50)

train_loader, test_loader = get_visco_loader(batch_size)

function RNO()
    return createRNO(1, 1, size(first(train_loader)[2], 1)) |> gpu
end

function Transformer()
    return createTransformer(1, 1, size(first(train_loader)[2], 1)) |> gpu
end

function KAN_RNO()
    return create_KAN_RNO(1, 1, size(first(train_loader)[2], 1), hyperparams, true) |> gpu
end

function KAN_Transformer()
    encoder_wavelet_names, decoder_wavelet_names, output_wavelet = hyperparams
    return create_KAN_Transformer(encoder_wavelet_names, decoder_wavelet_names, true, true, output_wavelet, true) |> gpu
end

instantiate_model = Dict(
    "RNO" => createRNO,
    "Transformer" => createTransformer,
    "KAN_RNO" => create_KAN_RNO,
    "KAN_Transformer" => create_KAN_Transformer
)[model_name]

log_file_base = Dict(
    "RNO" => "Vanilla_RNO/logs/",
    "Transformer" => "Vanilla_Transformer/logs/",
    "KAN_RNO" => "wavKAN_RNO/logs/",
    "KAN_Transformer" => "wavKAN_Transformer/logs/",
)[model_name]

for num in 1:NUM_REPETITIONS
    file_name = log_file_base * "repetition_" * string(num) * ".csv"
    seed = Random.seed!(num)
    model = instantiate_model()
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)

    train_loss = 0.0
    test_loss = 0.0

    start_time = time()
    for epoch in ProgressBar(1:num_epochs)
        model, opt_state, train_loss, test_loss = train_step(model, opt_state, train_loader, test_loader, loss_fcn, epoch)
        BIC = BIC(model, first(train_loader)[2], test_loss)
        time_epoch = time() - start_time
        log_csv(file_name, epoch, train_loss, test_loss, BIC, time_epoch)        
    end

    save_file_name = log_file_base * "trained_models/model_" * string(num) * ".bson"
    
    model = model |> cpu
    @save save_file_name model
end
