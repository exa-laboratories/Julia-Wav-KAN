using CSV, DataFrames, Statistics, Printf, PlotlyJS
using PlotlyJS: box, plot

log_locations = [
    "Vanilla_RNO/logs",
    # "wavKAN_RNO/logs",
    "Vanilla_Transformer/logs",
    "wavKAN_Transformer/logs",
]

plot_names = [
    "MLP RNO",
    #"wavKAN RNO",
    "MLP Transformer",
    "wavKAN Transformer",
]

num_repetitions = 5 

# Create an empty DataFrame to hold all results
results = DataFrame(Model = String[], train_loss = String[], test_loss = String[], BIC = String[])

box_plot_train = DataFrame(model = String[], value = Float64)
box_plot_test = DataFrame(model = String[], value = Float64)
box_plot_BIC = DataFrame(model = String[], value = Float64)

for (idx, log_location) in enumerate(log_locations)
    train_loss, test_loss, BIC = [], [], []
    for i in 1:num_repetitions
        df = CSV.read("$log_location/repetition_$i.csv", DataFrame)
        push!(train_loss, df[!,"Train Loss"][end])
        push!(test_loss, df[!,"Test Loss"][end])
        push!(BIC, df[!,"BIC"][end])

        push!(box_plot_train, (model = plot_names[idx], value = df[!,"Train Loss"][end]), promote = true)
        push!(box_plot_test, (model = plot_names[idx], value = df[!,"Test Loss"][end]), promote = true)
        push!(box_plot_BIC, (model = plot_names[idx], value = df[!,"BIC"][end]), promote = true)
    end

    # Calculate means and stds
    train_loss_mean = mean(train_loss)
    train_loss_std = std(train_loss)
    test_loss_mean = mean(test_loss)
    test_loss_std = std(test_loss)
    BIC_mean = mean(BIC)
    BIC_std = std(BIC)

    # Add results to the DataFrame
    push!(results, (
        Model = plot_names[idx],
        train_loss = @sprintf("%.2g ± %.2g", train_loss_mean, train_loss_std),
        test_loss = @sprintf("%.2g ± %.2g", test_loss_mean, test_loss_std),
        BIC = @sprintf("%.2g ± %.2g", BIC_mean, BIC_std)
    ))
end

headerColor = "grey"
rowEvenColor = "lightgrey"
rowOddColor = "white"

table_plot = plot(
    table(
        header = attr(values = ["Model", "Train Loss", "Test Loss", "BIC"],
        align="center",
        line_color="darkslategray",
        fill_color=headerColor,
        font=attr(family="Computer Modern", color="white", size=13)),
        cells = attr(values = [plot_names, results.train_loss, results.test_loss, results.BIC],
        line_color="darkslategray",
        fill_color=[ [rowOddColor,rowEvenColor,rowOddColor,rowEvenColor]],#,rowOddColor] ],
        align = "center",
        font = attr(family="Computer Modern", size=12, color="black")),
    ),
    Layout(
        autosize=true,
        title = attr(text = "Loss and BIC for Different Models", x = 0.5),
        font = attr(family="Computer Modern", size=12, color="black"),
        margin = attr(b = 0, t=200, l=5, r=5),
        scale=5,
    )
)

savefig(table_plot, "figures/loss_table.png")

# Create box plots
function box_data(df, name)
    data = []
    for (idx, plot_name) in enumerate(plot_names)
        vals = Float64.(df[df.model .== plot_name, :value])
        println(vals)
        trace = box(;y=vals, name=plot_name)
        push!(data, trace)
    end
    data = [data...]
    boxplot = plot(data)
    
    savefig(boxplot, "figures/$(name).png")
end


box_data(box_plot_train, "Train Loss")
box_data(box_plot_test, "Test Loss")
box_data(box_plot_BIC, "BIC")





