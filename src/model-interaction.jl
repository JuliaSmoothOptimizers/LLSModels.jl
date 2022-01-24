function NLPModelsModifiers.hess_structure!(
  nlp::FeasibilityFormNLS{T, S, LLSModel{T, S, M1, M2}},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1 <: AbstractMatrix, M2 <: AbstractMatrix}
  @lencheck nlp.meta.nnzh rows cols
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  rows .= (n + 1):(n + ne)
  cols .= (n + 1):(n + ne)
  return rows, cols
end

function NLPModelsModifiers.hess_coord!(
  nlp::FeasibilityFormNLS{T, S, LLSModel{T, S, M1, M2}},
  xr::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) where {T, S, M1 <: AbstractMatrix, M2 <: AbstractMatrix}
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  vals .= obj_weight
  return vals
end
