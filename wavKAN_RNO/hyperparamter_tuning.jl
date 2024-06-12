include("../pipeline/data_processing/data_loader.jl")
include("KAN_RNO.jl")
include("../utils.jl")
include("../pipeline/train.jl")

using HyperTuning
using ConfParser
using Random
using Flux, CUDA, KernelAbstractions
using Optimisers
using .training: train_step
using .KAN_RecurrentNO: create_KAN_RNO
using .loaders: get_visco_loader
using .UTILS: loss_fcn, BIC

# Define the objective function, edits RNO_config.ini and runs the training 
function objective(trial)
    seed = get_seed(trial)
    Random.seed!(seed)

    @suggest n_hidden in trial
    @suggest n_layers in trial
    @suggest activation in trial
    @suggest b_size in trial
    @suggest learning_rate in trial
    @suggest gamma in trial
    @suggest step_rate in trial
    @suggest wav_one in trial
    @suggest wav_two in trial
    @suggest wav_three in trial
    @suggest wav_four in trial
    @suggest wav_five in trial

    wavelet_names = [wav_one, wav_two, wav_three, wav_four, wav_five][1:n_layers]

    # Parse config
    conf = ConfParse("wavKAN_RNO/KAN_RNO_config.ini")
    parse_conf!(conf)

    # Create model
    ENV["p"] = retrieve(conf, "Loss", "p")
    ENV["step"] = step_rate
    ENV["decay"] = gamma
    ENV["LR"] = learning_rate
    ENV["min_LR"] = retrieve(conf, "Optimizer", "min_lr")
    ENV["activation"] = activation
    ENV["n_hidden"] = n_hidden
    ENV["num_layers"] = n_layers

    train_loader, test_loader = get_visco_loader(b_size)

    model = create_KAN_RNO(1, 1, size(first(train_loader)[2], 1), wavelet_names, true) |> gpu

    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)

    train_loss = 0.0
    test_loss = 0.0
    num_epochs = parse(Int, retrieve(conf, "Pipeline", "num_epochs"))
    for epoch in 1:num_epochs
        model, opt_state, train_loss, test_loss = train_step(model, opt_state, train_loader, test_loader, loss_fcn, epoch)
    end

    println("Train Loss: ", train_loss, " Test Loss: ", test_loss)

    # Return BIC
    return BIC(model, first(train_loader)[2], test_loss)

end

wavelet_list = ["MexicanHat", "Morlet", "DerivativeOfGaussian", "Shannon", "Meyer"]

# Define the search space
space = Scenario(
    n_hidden = 2:20,
    n_layers = 2:5,
    wav_one = wavelet_list,
    wav_two = wavelet_list,
    wav_three = wavelet_list,
    wav_four = wavelet_list,
    wav_five = wavelet_list,
    activation = ["relu", "selu", "leakyrelu"],
    b_size = 1:20,
    learning_rate = (1e-4..1e-1),
    gamma = (0.1..0.9),
    step_rate = 10:40,
    verbose = true,
    max_trials = 50,
    pruner = MedianPruner(start_after = 5, prune_after = 10),
)

HyperTuning.optimize(objective, space)

display(top_parameters(space))

# Save the best configuration
@unpack n_hidden, n_layers, activation, b_size, learning_rate, gamma, step_rate, wav_one, wav_two, wav_three, wav_four, wav_five = space

conf = ConfParse("wavKAN_RNO/KAN_RNO_config.ini")
parse_conf!(conf)

commit!(conf, "Architecture", "n_hidden", string(n_hidden))
commit!(conf, "Architecture", "n_layers", string(n_layers))
commit!(conf, "Architecture", "activation", string(activation))
commit!(conf, "Architecture", "wav_one", string(wav_one))
commit!(conf, "Architecture", "wav_two", string(wav_two))
commit!(conf, "Architecture", "wav_three", string(wav_three))
commit!(conf, "Architecture", "wav_four", string(wav_four))
commit!(conf, "Architecture", "wav_five", string(wav_five))
commit!(conf, "DataLoader", "batch_size", string(b_size))
commit!(conf, "Optimizer", "learning_rate", string(learning_rate))
commit!(conf, "Optimizer", "gamma", string(gamma))
commit!(conf, "Optimizer", "step_rate", string(step_rate))

save!(conf, "wavKAN_RNO/KAN_RNO_config.ini")












