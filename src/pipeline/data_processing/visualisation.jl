using Plots; pythonplot()
using CUDA
using Flux

include("./data_loader.jl")
using .loaders: get_visco_loader

train_loader, test_loader = get_visco_loader(1)

epsi_first, sigma_first = first(test_loader) |> cpu
num_samples = size(epsi_first, 1)

anim = @animate for i in 1:num_samples
    epsi = epsi_first[1:i,1]
    sigma = sigma_first[1:i,1]  
    plot(epsi, sigma, title="True Viscoplastic Data", xlabel="Strain", ylabel="Stress", color=:blue, label="Test Sample 1")
    xlims!(0, 1)
    ylims!(0, 1)
end

# Save the animation to file
gif(anim, "figures/true_test_visco_data.gif", fps=30)

epsi_first, sigma_first = first(train_loader) |> cpu
num_samples = size(epsi_first, 1)

anim = @animate for i in 1:num_samples
    epsi = epsi_first[1:i,1]
    sigma = sigma_first[1:i,1]  
    plot(epsi, sigma, title="True Viscoplastic Data", xlabel="Strain", ylabel="Stress", color=:red, label="Train Sample 1")
    xlims!(0, 1)
    ylims!(0, 1)
end

# Save the animation to file
gif(anim, "figures/true_train_visco_data.gif", fps=30)