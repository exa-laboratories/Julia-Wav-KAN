include("./mexican_hat.jl")

using ConfParser
using Plots; pythonplot()
using .MexicanHat: MexicanHatWavelet
using Printf    

# Parse config
conf = ConfParse("wavelet_config.ini")
parse_conf!(conf)

wavelet = "MexicanHat"
wavelet_transform = Dict(
    "MexicanHat" => MexicanHatWavelet
)[wavelet]

σ_list = range(0.1, 5, length=100)
x = range(-15, 15, length=100)

wavelet_gif = @animate for σ in σ_list
    label = @sprintf("σ = %.2f", σ)
    y = wavelet_transform(σ).(x)
    plot(x, y, label=label, xlabel="x", ylabel="ψ(x)", title=wavelet *" Wavelet", color=:yellow, background_color=:black, grid=true, color_palette=:grays, gridalpha=0.5, legendfontsize=12)
    xlims!(x[1], x[end])
    ylims!(-1, 1)
end

gif(wavelet_gif, "wavelets/animations/$wavelet.gif", fps=20)
