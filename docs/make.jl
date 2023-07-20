using Documenter, LLSModels

makedocs(
  modules = [LLSModels],
  doctest = true,
  linkcheck = false,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "LLSModels.jl",
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/LLSModels.jl.git",
  push_preview = true,
  devbranch = "main",
)
