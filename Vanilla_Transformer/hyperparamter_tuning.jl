include("../pipeline/data_processing/data_loader.jl")
include("Transformer.jl")
include("../utils.jl")
include("../pipeline/train.jl")

using HyperTuning
using ConfParser
using Random
using Flux, CUDA, KernelAbstractions
using Optimisers
using .training: train_step
using .TransformerModel: createTransformer
using .loaders: get_visco_loader
using .UTILS: loss_fcn

# Define the objective function, edits RNO_config.ini and runs the training 
function objective(trial)
    seed = get_seed(trial)
    Random.seed!(seed)

    @suggest d_model in trial
    @suggest nhead in trial
    @suggest dim_feedforward in trial
    @suggest dropout in trial
    @suggest num_encoder_layers in trial
    @suggest num_decoder_layers in trial
    @suggest max_len in trial
    @suggest activation in trial
    @suggest b_size in trial
    @suggest learning_rate in trial
    @suggest gamma in trial
    @suggest step_rate in trial

    # Parse config
    conf = ConfParse("Vanilla_Transformer/Transformer_config.ini")
    parse_conf!(conf)

    # Create model
    ENV["p"] = retrieve(conf, "Loss", "p")
    ENV["step"] = step_rate
    ENV["decay"] = gamma
    ENV["LR"] = learning_rate
    ENV["min_LR"] = retrieve(conf, "Optimizer", "min_lr")
    ENV["activation"] = activation
    ENV["d_model"] = d_model
    ENV["nhead"] = nhead
    ENV["dim_feedforward"] = dim_feedforward
    ENV["dropout"] = dropout
    ENV["num_encoder_layers"] = num_encoder_layers
    ENV["num_decoder_layers"] = num_decoder_layers
    ENV["max_len"] = max_len

    train_loader, test_loader = get_visco_loader(b_size)

    model = createTransformer() |> gpu

    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)

    train_loss = 0.0
    test_loss = 0.0
    num_epochs = parse(Int, retrieve(conf, "Pipeline", "num_epochs"))
    for epoch in 1:num_epochs
        model, opt_state, train_loss, test_loss = train_step(model, opt_state, train_loader, test_loader, loss_fcn, epoch)
        report_value!(trial, test_loss)
        should_prune(trial) && (return)
    end

    test_loss < 100 && report_success!(trial)

end

# Define the search space
space = Scenario(
    d_model = range(64, 192, step=2),
    nhead = 1:20,
    dim_feedforward = 500:1200,
    dropout = (0.1..0.9),
    num_encoder_layers = 2:8,
    num_decoder_layers = 1:3,
    max_len = 1000:5000,
    activation = ["relu", "selu", "leakyrelu", "swish", "gelu"],
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
@unpack d_model, nhead, dim_feedforward, dropout, num_encoder_layers, num_decoder_layers, max_len, activation, b_size, learning_rate, gamma, step_rate = space

conf = ConfParse("Vanilla_Transformer/Transformer_config.ini")
parse_conf!(conf)

commit!(conf, "Architecture", "d_model", string(d_model))
commit!(conf, "Architecture", "nhead", string(nhead))
commit!(conf, "Architecture", "dim_feedforward", string(dim_feedforward))
commit!(conf, "Architecture", "dropout", string(dropout))
commit!(conf, "Architecture", "num_encoder_layers", string(num_encoder_layers))
commit!(conf, "Architecture", "num_decoder_layers", string(num_decoder_layers))
commit!(conf, "Architecture", "max_len", string(max_len))
commit!(conf, "Architecture", "activation", string(activation))
commit!(conf, "DataLoader", "batch_size", string(b_size))
commit!(conf, "Optimizer", "learning_rate", string(learning_rate))
commit!(conf, "Optimizer", "gamma", string(gamma))
commit!(conf, "Optimizer", "step_rate", string(step_rate))

save!(conf, "Vanilla_Transformer/Transformer_config.ini")








