module MATFileLoader

export load_visco_data

using MAT

function load_visco_data()
    
    N_total = parse(Int, get(ENV, "N_total", "400"))
    N_train = parse(Int, get(ENV, "N_train", "300"))
    N_test = N_total - N_train

    matfile = matread("1D_Viscoplastic_Data/viscodata_3mat.mat")

    epsi_field = Float32.(matfile["epsi_tol"])[1:N_total, :]
    sigma_field = Float32.(matfile["sigma_tol"])[1:N_total, :]

    # Split data
    epsi_train = epsi_field[1:N_train, :]
    sigma_train = sigma_field[1:N_train, :]
    epsi_test = epsi_field[N_train+1:end, :]
    sigma_test = sigma_field[N_train+1:end, :]

    return epsi_train, sigma_train, epsi_test, sigma_test
end
end