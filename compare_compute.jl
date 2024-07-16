include("src/pipeline/data_processing/data_loader.jl")
include("src/models/Vanilla_RNO/RNO.jl")
include("src/models/Vanilla_Transformer/Transformer.jl")
include("src/models/wavKAN_RNO/KAN_RNO.jl")
include("src/models/wavKAN_Transformer/KAN_Transformer.jl")

using Flux
using BSON: @load
using CUDA, KernelAbstractions
using .loaders: get_visco_loader
using ConfParser
using GFlops # Not currently supported in Julia 1.10, will implement this file properly in the future

train_loader, test_loader = get_visco_loader(1)

MODEL_NAME = "Transformer"

model_file = Dict(
    "RNO" => "src/models/Vanilla_RNO/logs/trained_models/model_1.bson",
    "KAN_RNO" => "src/models/wavKAN_RNO/logs/trained_models/model_1.bson",
    "Transformer" => "src/models/Vanilla_Transformer/logs/trained_models/model_1.bson",
    "KAN_Transformer" => "src/models/wavKAN_Transformer/logs/trained_models/model_1.bson"
)[MODEL_NAME]
    
# Load the model
@load model_file model
model = model |> gpu

epsi_first, sigma_first = first(test_loader)
num_samples = size(epsi_first, 1)

flops = @gflops model(epsi_first, sigma_first)

println("GFLOPS: $flops")