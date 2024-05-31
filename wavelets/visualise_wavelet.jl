include("./mexican_hat.jl")
include("./morlet.jl")
include("./derivative_of_gaussian.jl")
include("./shannon.jl")
include("./meyer.jl")

using Plots; pythonplot()
using .MexicanHat: MexicanHatWavelet
using .Morlet: MorletWavelet
using .DoG: DoGWavelet
using .Shannon: ShannonWavelet
using .Meyer: MeyerWavelet
using Printf    
using LinearAlgebra

wavelet = "Meyer"
wavelet_transform = Dict(
    "MexicanHat" => MexicanHatWavelet,
    "Morlet" => MorletWavelet,
    "DerivativeOfGaussian" => DoGWavelet,
    "Shannon" => ShannonWavelet,
    "Meyer" => MeyerWavelet
)[wavelet]

σ_list = range(0.1, 5, length=100)
γ_list = range(1, 50, length=100)
b = 0
x = range(-15, 15, length=100)

args = Dict(
    "MexicanHat" => σ_list,
    "Morlet" => γ_list,
    "DerivativeOfGaussian" => σ_list,
    "Shannon" => [(σ, y) for σ in σ_list, y in range(-5, 5, length=10)],
    "Meyer" => [(σ, y) for σ in σ_list, y in range(-5, 5, length=10)]
)[wavelet]

symbol = Dict(
    "MexicanHat" => "σ",
    "Morlet" => "γ",
    "DerivativeOfGaussian" => "σ",
    "Shannon" => "σ",
    "Meyer" => "σ"
)[wavelet]

weights = Matrix{Float32}(I, 100, 100)

wavelet_gif = @animate for arg in args
    y = wavelet_transform(arg..., weights)(x)
    if wavelet == "Shannon" || wavelet == "Meyer"
        arg = arg[1]
    end
    label = @sprintf("%s = %.2f", symbol, arg)
    plot(x, y, label=label, xlabel="x", ylabel="ψ(x)", title=wavelet *" Wavelet", color=:yellow, background_color=:black, grid=true, color_palette=:grays, gridalpha=0.5, legendfontsize=12)
    xlims!(x[1], x[end])
    ylims!(-1, 1)
end

gif(wavelet_gif, "wavelets/animations/$wavelet.gif", fps=20)
