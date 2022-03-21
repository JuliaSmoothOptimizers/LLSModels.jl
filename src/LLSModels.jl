module LLSModels

using LinearAlgebra, SparseArrays
using LinearOperators, NLPModels, NLPModelsModifiers, SparseMatricesCOO

include("lls_model.jl")
include("model-interaction.jl")
include("solve_krylov.jl")

end # module
