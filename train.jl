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

model_name = "KAN_Transformer"
hparams = set_hyperparams(model_name)
batch_size = parse(Int, get(ENV, "batch_size", "32"))
learning_rate = parse(Float32, get(ENV, "LR", "1e-3"))
num_epochs = parse(Int, get(ENV, "num_epochs", "50"))
optimizer_name = get(ENV, "optimizer", "Adam")
norm = parse(Bool, get(ENV, "norm", "false"))

train_loader, test_loader = get_visco_loader(batch_size)

function RNO()
    return createRNO(1, 1, size(first(train_loader)[2], 1)) |> gpu
end

function Transformer()
    return createTransformer() |> gpu
end

function KAN_RNO()
    return create_KAN_RNO(1, 1, size(first(train_loader)[2], 1), hparams, norm) |> gpu
end

function KAN_Transformer()
    encoder_wavelet_names, decoder_wavelet_names, output_wavelet = hparams
    return create_KAN_Transformer(encoder_wavelet_names, decoder_wavelet_names, norm, norm, output_wavelet, norm) |> gpu
end

instantiate_model = Dict(
    "RNO" => RNO,
    "Transformer" => Transformer,
    "KAN_RNO" => KAN_RNO,
    "KAN_Transformer" => KAN_Transformer
)[model_name]

log_file_base = Dict(
    "RNO" => "Vanilla_RNO/logs/",
    "Transformer" => "Vanilla_Transformer/logs/",
    "KAN_RNO" => "wavKAN_RNO/logs/",
    "KAN_Transformer" => "wavKAN_Transformer/logs/",
)[model_name]

optimizer = Dict(
    "Adam" => Optimisers.Adam(learning_rate),
    "SGD" => Optimisers.Descent(learning_rate)
)[optimizer_name]

for num in 1:NUM_REPETITIONS
    file_name = log_file_base * "repetition_" * string(num) * ".csv"
    seed = Random.seed!(num)
    model = instantiate_model()
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)

    train_loss = 0.0
    test_loss = 0.0

    # Create csv with header
    open(file_name, "w") do file
        write(file, "Epoch,Time (s),Train Loss,Test Loss,BIC\n")
    end

    start_time = time()
    for epoch in ProgressBar(1:num_epochs)
        model, opt_state, train_loss, test_loss = train_step(model, opt_state, train_loader, test_loader, loss_fcn, epoch)
        BIC_val = BIC(model, first(train_loader)[2], test_loss)
        time_epoch = time() - start_time
        log_csv(epoch, train_loss, test_loss, BIC_val, time_epoch, file_name)  
    end

    save_file_name = log_file_base * "trained_models/model_" * string(num) * ".bson"
    
    model = model |> cpu
    @save save_file_name model

    model = nothing
end
