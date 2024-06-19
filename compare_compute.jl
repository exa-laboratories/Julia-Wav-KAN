include("pipeline/data_processing/data_loader.jl")
include("Vanilla_RNO/RNO.jl")
include("Vanilla_Transformer/Transformer.jl")
include("wavKAN_RNO/KAN_RNO.jl")
include("wavKAN_Transformer/KAN_Transformer.jl")

using Flux
using BSON: @load
using CUDA, KernelAbstractions
using .loaders: get_visco_loader
using ConfParser
using GFlops # Not currently supported in Julia 1.10, will implement this file properly in the future

train_loader, test_loader = get_visco_loader(1)

MODEL_NAME = "Transformer"

model_file = Dict(
    "RNO" => "Vanilla_RNO/logs/trained_models/model_1.bson",
    "KAN_RNO" => "wavKAN_RNO/logs/trained_models/model_1.bson",
    "Transformer" => "Vanilla_Transformer/logs/trained_models/model_1.bson",
    "KAN_Transformer" => "wavKAN_Transformer/logs/trained_models/model_1.bson"
)[MODEL_NAME]
    
# Load the model
@load model_file model
model = model |> gpu

epsi_first, sigma_first = first(test_loader)
num_samples = size(epsi_first, 1)

flops = @gflops model(epsi_first, sigma_first)

println("GFLOPS: $flops")