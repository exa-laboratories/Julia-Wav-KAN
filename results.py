import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

log_locations = [
    "Vanilla_RNO/logs",
    "wavKAN_RNO/logs",
    "Vanilla_Transformer/logs",
    "wavKAN_Transformer/logs",
]

plot_names = [
    "MLP RNO",
    "wavKAN RNO",
    "MLP Transformer",
    "wavKAN Transformer",
]

model_file = [
    "Vanilla_RNO/logs/trained_models/model_5.bson",
    "wavKAN_RNO/logs/trained_models/model_1.bson",  # This is the best one
    "Vanilla_Transformer/logs/trained_models/model_3.bson", 
    "wavKAN_Transformer/logs/trained_models/model_2.bson"  # This is the best one
]

# This array for params counts has been generated by sum(length, Flux.params(model)) for each model
param_counts = [52, 4413, 4209205, 489562]

num_repetitions = 5 

# Create an empty DataFrame to hold all results
results = pd.DataFrame(columns=["Model", "train_loss", "test_loss", "BIC", "time", "param_count"])

box_plot_train = pd.DataFrame(columns=["model", "value"])
box_plot_test = pd.DataFrame(columns=["model", "value"])
box_plot_BIC = pd.DataFrame(columns=["model", "value"])
box_plot_time = pd.DataFrame(columns=["model", "value"])

for idx, log_location in enumerate(log_locations):
    train_loss, test_loss, BIC, time = [], [], [], []
    for i in range(1, num_repetitions + 1):
        df = pd.read_csv(f"{log_location}/repetition_{i}.csv")
        if pd.isna(df["Test Loss"].iloc[-1]):
            continue
        train_loss.append(df["Train Loss"].iloc[-1])
        test_loss.append(df["Test Loss"].iloc[-1])
        BIC.append(df["BIC"].iloc[-1])
        time.append(df["Time (s)"].iloc[-1] / 60)

        box_plot_train = pd.concat([box_plot_train, pd.DataFrame({"model": [plot_names[idx]], "value": [df["Train Loss"].iloc[-1]]})])
        box_plot_test = pd.concat([box_plot_test, pd.DataFrame({"model": [plot_names[idx]], "value": [df["Test Loss"].iloc[-1]]})])
        box_plot_BIC = pd.concat([box_plot_BIC, pd.DataFrame({"model": [plot_names[idx]], "value": [df["BIC"].iloc[-1]]})])
        box_plot_time = pd.concat([box_plot_time, pd.DataFrame({"model": [plot_names[idx]], "value": [df["Time (s)"].iloc[-1] / 60]})])

    results = pd.concat([results, pd.DataFrame({
        "Model": [plot_names[idx]],
        "train_loss": [f"{np.mean(train_loss):.2g} ± {np.std(train_loss):.2g}"],
        "test_loss": [f"{np.mean(test_loss):.2g} ± {np.std(test_loss):.2g}"],
        "BIC": [f"{np.mean(BIC):.2g} ± {np.std(BIC):.2g}"],
        "time": [f"{np.mean(time):.2g} ± {np.std(time):.2g}"],
        "param_count": [param_counts[idx]]
    })])

# Create a table
header = ["Model", "Train Loss", "Test Loss", "BIC", "Time (mins)", "Param Count"]
table = tabulate(results.values, headers=header, tablefmt="grid")
print(table)

# Save the table as a text file
with open("figures/loss_table.txt", "w") as f:
    f.write(table)

def box_data(df, name):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="model", y="value", data=df)
    plt.title(name)
    plt.xlabel("Model")
    plt.ylabel(name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png")
    plt.close()

box_data(box_plot_train, "Train Loss")
box_data(box_plot_test, "Test Loss")
box_data(box_plot_BIC, "BIC")
box_data(box_plot_time, "Time (mins)")
