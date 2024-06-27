include("../pipeline/data_processing/data_loader.jl")
include("KAN_Transformer.jl")
include("../utils.jl")
include("../pipeline/train.jl")
include("../hp_parsing.jl")

using HyperTuning
using ConfParser
using Random
using Flux, CUDA, KernelAbstractions
using Optimisers
using .training: train_step
using .KANTransformerModel: create_KAN_Transformer
using .loaders: get_visco_loader
using .UTILS: loss_fcn
using .hyperparams: set_hyperparams

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
    @suggest encoder_wav_one in trial
    @suggest encoder_wav_two in trial
    @suggest encoder_wav_three in trial
    @suggest encoder_wav_four in trial
    @suggest encoder_wav_five in trial
    @suggest encoder_wav_six in trial
    @suggest encoder_wav_seven in trial
    @suggest encoder_wav_eight in trial
    @suggest num_decoder_layers in trial
    @suggest decoder_wav_one in trial
    @suggest decoder_wav_two in trial
    @suggest decoder_wav_three in trial
    @suggest output_wavelet in trial

    # Parse config
    conf = ConfParse("wavKAN_Transformer/KAN_Transformer_config.ini")
    parse_conf!(conf)

    # Use Vanilla_Transformer config
    # _ = set_hyperparams("Transformer")
    # b_size = parse(Int, get(ENV, "batch_size", "32"))
    # learning_rate = parse(Float32, get(ENV, "LR", "1e-3"))
    num_epochs = 15

    num_encoder_layers = parse(Int, get(ENV, "num_encoder_layers", "2"))
    num_decoder_layers = parse(Int, get(ENV, "num_decoder_layers", "2"))

    encoder_wavelet_names = [encoder_wav_one, encoder_wav_two, encoder_wav_three, encoder_wav_four, encoder_wav_five, encoder_wav_six, encoder_wav_seven, encoder_wav_eight][1:num_encoder_layers]
    decoder_wavelet_names = [decoder_wav_one, decoder_wav_two, decoder_wav_three][1:num_decoder_layers]

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

    model = create_KAN_Transformer(encoder_wavelet_names, decoder_wavelet_names, true, true, output_wavelet, true) |> gpu

    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)

    train_loss = 0.0
    test_loss = 0.0
    # num_epochs = parse(Int, retrieve(conf, "Pipeline", "num_epochs"))
    for epoch in 1:num_epochs
        model, opt_state, train_loss, test_loss = train_step(model, opt_state, train_loader, test_loader, loss_fcn, epoch)
        report_value!(trial, test_loss)
        should_prune(trial) && (return)
    end

    model = nothing
    train_loader = nothing
    test_loader = nothing
    GC.gc(true) 
    CUDA.reclaim()

    test_loss < 100 && report_success!(trial)
    return test_loss

end

wavelet_list = ["MexicanHat", "DerivativeOfGaussian", "Morlet", "Shannon", "Meyer"]


# Define the search space
space = Scenario(
    d_model = range(10, 74, step=2),
    nhead = 1:7,
    dim_feedforward = 300:750,
    dropout = (0.1..0.9),
    num_encoder_layers = 2:6,
    encoder_wav_one = wavelet_list,
    encoder_wav_two = wavelet_list,
    encoder_wav_three = wavelet_list,
    encoder_wav_four = wavelet_list,
    encoder_wav_five = wavelet_list,
    encoder_wav_six = wavelet_list,
    encoder_wav_seven = wavelet_list,
    encoder_wav_eight = wavelet_list,
    num_decoder_layers = [1,1],
    decoder_wav_one = wavelet_list,
    decoder_wav_two = wavelet_list,
    decoder_wav_three = wavelet_list,
    output_wavelet = wavelet_list,
    max_len = 251:450,
    activation = ["relu", "selu", "leakyrelu", "swish", "gelu"],
    b_size = 1:5,
    learning_rate = (1e-6..1e-1),
    gamma = (0.5..0.9),
    step_rate = 10:40,
    verbose = true,
    max_trials = 100,
    pruner = MedianPruner(),
)

HyperTuning.optimize(objective, space)

display(top_parameters(space))

# Save the best configuration
@unpack d_model, nhead, dim_feedforward, dropout, num_encoder_layers, num_decoder_layers, max_len, activation, b_size, learning_rate, gamma, step_rate, encoder_wav_one, encoder_wav_two, encoder_wav_three, encoder_wav_four, encoder_wav_five, encoder_wav_six, encoder_wav_seven, encoder_wav_eight, decoder_wav_one, decoder_wav_two, decoder_wav_three, output_wavelet = space

conf = ConfParse("wavKAN_Transformer/KAN_Transformer_config.ini")
parse_conf!(conf)

vanilla_conf = ConfParse("Vanilla_Transformer/Transformer_config.ini")
parse_conf!(vanilla_conf)

# # Take vanilla config 
# d_model = retrieve(vanilla_conf, "Architecture", "d_model")
# nhead = retrieve(vanilla_conf, "Architecture", "nhead")
# dim_feedforward = retrieve(vanilla_conf, "Architecture", "dim_feedforward")
# dropout = retrieve(vanilla_conf, "Architecture", "dropout")
# max_len = retrieve(vanilla_conf, "Architecture", "max_len")
# activation = retrieve(vanilla_conf, "Architecture", "activation")
# b_size = retrieve(vanilla_conf, "DataLoader", "batch_size")
# learning_rate = retrieve(vanilla_conf, "Optimizer", "learning_rate")
# gamma = retrieve(vanilla_conf, "Optimizer", "gamma")
# step_rate = retrieve(vanilla_conf, "Optimizer", "step_rate")

commit!(conf, "Architecture", "d_model", string(d_model))
commit!(conf, "Architecture", "nhead", string(nhead))
commit!(conf, "Architecture", "dim_feedforward", string(dim_feedforward))
commit!(conf, "Architecture", "dropout", string(dropout))
commit!(conf, "Architecture", "num_encoder_layers", string(num_encoder_layers))
commit!(conf, "Architecture", "num_decoder_layers", string(num_decoder_layers))
commit!(conf, "Architecture", "max_len", string(max_len))
commit!(conf, "Architecture", "activation", string(activation))
commit!(conf, "EncoderWavelets", "wav_one", encoder_wav_one)
commit!(conf, "EncoderWavelets", "wav_two", encoder_wav_two)
commit!(conf, "EncoderWavelets", "wav_three", encoder_wav_three)
commit!(conf, "EncoderWavelets", "wav_four", encoder_wav_four)
commit!(conf, "EncoderWavelets", "wav_five", encoder_wav_five)
commit!(conf, "EncoderWavelets", "wav_six", encoder_wav_six)
commit!(conf, "EncoderWavelets", "wav_seven", encoder_wav_seven)
commit!(conf, "EncoderWavelets", "wav_eight", encoder_wav_eight)
commit!(conf, "DecoderWavelets", "wav_one", decoder_wav_one)
commit!(conf, "DecoderWavelets", "wav_two", decoder_wav_two)
commit!(conf, "DecoderWavelets", "wav_three", decoder_wav_three)
commit!(conf, "OutputWavelet", "wav", output_wavelet)
commit!(conf, "DataLoader", "batch_size", string(b_size))
commit!(conf, "Optimizer", "learning_rate", string(learning_rate))
commit!(conf, "Optimizer", "gamma", string(gamma))
commit!(conf, "Optimizer", "step_rate", string(step_rate))

save!(conf, "wavKAN_Transformer/KAN_Transformer_config.ini")








