include("./mexican_hat.jl")
include("./morlet.jl")

using Plots; pythonplot()
using .MexicanHat: MexicanHatWavelet
using .Morlet: MorletWavelet
using Printf    

wavelet = "Morlet"
wavelet_transform = Dict(
    "MexicanHat" => MexicanHatWavelet,
    "Morlet" => MorletWavelet
)[wavelet]

σ_list = range(0.1, 5, length=100)
γ_list = range(1, 50, length=100)
x = range(-15, 15, length=100)

args = Dict(
    "MexicanHat" => σ_list,
    "Morlet" => γ_list
)[wavelet]

symbol = Dict(
    "MexicanHat" => "σ",
    "Morlet" => "γ"
)[wavelet]

weights = [1.0 for _ in length(args)]

wavelet_gif = @animate for arg in args
    y = wavelet_transform(arg..., weights)(x)
    label = @sprintf("%s = %.2f", symbol, arg)
    plot(x, y, label=label, xlabel="x", ylabel="ψ(x)", title=wavelet *" Wavelet", color=:yellow, background_color=:black, grid=true, color_palette=:grays, gridalpha=0.5, legendfontsize=12)
    xlims!(x[1], x[end])
    ylims!(-1, 1)
end

gif(wavelet_gif, "wavelets/animations/$wavelet.gif", fps=20)
