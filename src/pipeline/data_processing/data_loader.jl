module loaders

include("./data_reader.jl")
include("../../utils.jl")
using .MATFileLoader: load_visco_data
using .UTILS: UnitGaussianNormaliser, unit_encode, MinMaxNormaliser, minmax_encode
using Flux
using CUDA

function get_visco_loader(batch_size=32)
    epsi_train, sigma_train, epsi_test, sigma_test = load_visco_data()

    epsi_normaliser = MinMaxNormaliser(epsi_train)
    sigma_normaliser = MinMaxNormaliser(sigma_train)

    # Down-sample the data to a coarser grid in time - reduces the training time
    s = 4
    epsi_train = epsi_train[:, 1:s:end]
    sigma_train = sigma_train[:, 1:s:end]
    epsi_test = epsi_test[:, 1:s:end]
    sigma_test = sigma_test[:, 1:s:end]

    # Normalise
    epsi_train = minmax_encode(epsi_normaliser, epsi_train)
    sigma_train = minmax_encode(sigma_normaliser, sigma_train)
    epsi_test = minmax_encode(epsi_normaliser, epsi_test)
    sigma_test = minmax_encode(sigma_normaliser, sigma_test)

    epsi_train = permutedims(epsi_train, [2, 1])
    sigma_train = permutedims(sigma_train, [2, 1])
    epsi_test = permutedims(epsi_test, [2, 1])
    sigma_test = permutedims(sigma_test, [2, 1])

    train_loader = Flux.DataLoader((epsi_train, sigma_train) |> gpu, batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader((epsi_test, sigma_test) |> gpu, batchsize=batch_size, shuffle=false)

    return train_loader, test_loader
    
end
end

