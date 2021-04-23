function NLPModelsModifiers.hess_structure!(
  nlp::FeasibilityFormNLS{LLSModel},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nnzh rows cols
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  rows .= (n + 1):(n + ne)
  cols .= (n + 1):(n + ne)
  return rows, cols
end

function NLPModelsModifiers.hess_coord!(
  nlp::FeasibilityFormNLS{LLSModel},
  xr::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  vals .= obj_weight
  return vals
end
