include("pipeline/data_processing/data_loader.jl")
include("Vanilla_RNO/RNO.jl")
include("Vanilla_Transformer/Transformer.jl")
include("wavKAN_RNO/KAN_RNO.jl")
include("wavKAN_Transformer/KAN_Transformer.jl")

using Plots; pythonplot() # There's a clash with PlotlyJS here, so use Pkg.rm("PlotlyJS") if you want to plot predictions.
using Flux
using BSON: @load
using CUDA, KernelAbstractions
using .loaders: get_visco_loader
using ConfParser

train_loader, test_loader = get_visco_loader(1)

MODEL_NAME = "KAN_RNO"

model_file = Dict(
    "RNO" => "Vanilla_RNO/logs/trained_models/model_1.bson",
    "KAN_RNO" => "wavKAN_RNO/logs/trained_models/model_2.bson", # This is the best one
    "Transformer" => "Vanilla_Transformer/logs/trained_models/model_1.bson", 
    "KAN_Transformer" => "wavKAN_Transformer/logs/trained_models/model_2.bson" # This is the best one
)[MODEL_NAME]
    
# Load the model
@load model_file model
model = model |> gpu

epsi_first, sigma_first = first(test_loader)
num_samples = size(epsi_first, 1)

predicted_stress = model(epsi_first, sigma_first) 
predicted_stress = copy(predicted_stress) |> cpu

epsi_first, sigma_first = epsi_first |> cpu, sigma_first |> cpu

delay = 30

anim = @animate for i in 1:(num_samples + delay)
    if i <= num_samples && i <= delay
        epsi = epsi_first[1:i,1]
        sigma = sigma_first[1:i,1] 
        pred_epsi = [NaN]
        pred_sigma = [NaN]
    elseif i <= num_samples && i > delay
        epsi = epsi_first[1:i,1]
        sigma = sigma_first[1:i,1] 
        pred_epsi = epsi_first[1:i-delay,1]
        pred_sigma = predicted_stress[1:i-delay,1]
    else
        epsi = epsi_first[1:num_samples,1]
        sigma = sigma_first[1:num_samples,1] 
        pred_epsi = epsi_first[1:i-delay,1]
        pred_sigma = predicted_stress[1:i-delay,1]
    end
    plot([epsi, pred_epsi], [sigma, pred_sigma], title="$MODEL_NAME Test Sample Prediction", xlabel="Strain", ylabel="Stress", color=[:blue :red], label=["True" "$MODEL_NAME Predicted"])
    xlims!(0, 1)
    ylims!(0, 1)
end

# Save the animation to file
gif(anim, "figures/$MODEL_NAME" * "_visco_prediction.gif", fps=15)