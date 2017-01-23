using Documenter, NLPModels

makedocs(
  modules = [NLPModels]
)

deploydocs(deps = Deps.pip("pygments", "mkdocs", "mkdocs-material", "python-markdown-math"),
  repo = "github.com/JuliaSmoothOptimizers/NLPModels.jl.git",
  julia = "0.5",
  latest = "master"
)
