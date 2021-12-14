export LLSModel

"""
    nls = LLSModel(A, b; lvar, uvar, C, lcon, ucon)

Creates a Linear Least Squares model ``\\tfrac{1}{2}\\|Ax - b\\|^2`` with optional bounds
`lvar ≦ x ≦ uvar` and optional linear constraints `lcon ≦ Cx ≦ ucon`.
This problem is a nonlinear least-squares problem with residual given by ``F(x) = Ax - b``.
"""
mutable struct LLSModel{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters

  Arows::Vector{Int}
  Acols::Vector{Int}
  Avals::S
  b::S
  Crows::Vector{Int}
  Ccols::Vector{Int}
  Cvals::S
end

NLPModels.show_header(io::IO, nls::LLSModel) = println(io, "LLSModel - Linear least-squares model")

function LLSModel(
  A::AbstractMatrix,
  b::S;
  x0::S = fill!(S(undef, size(A, 2)), zero(eltype(b))),
  lvar::S = fill!(S(undef, size(A, 2)), eltype(b)(-Inf)),
  uvar::S = fill!(S(undef, size(A, 2)), eltype(b)(Inf)),
  C::AbstractMatrix = similar(b, 0, 0),
  lcon::S = S(undef, 0),
  ucon::S = S(undef, 0),
  y0::S = fill!(S(undef, size(C, 1)), zero(eltype(b))),
  name::String = "generic-LLSModel",
) where {S}
  nvar = size(A, 2)
  Arows, Acols, Avals = if A isa AbstractSparseMatrix
    findnz(A)
  else
    m, n = size(A)
    I = ((i, j) for i = 1:m, j = 1:n)
    getindex.(I, 1)[:], getindex.(I, 2)[:], A[:]
  end
  Crows, Ccols, Cvals = if C isa AbstractSparseMatrix
    findnz(C)
  else
    m, n = size(C)
    I = ((i, j) for i = 1:m, j = 1:n)
    getindex.(I, 1)[:], getindex.(I, 2)[:], C[:]
  end
  LLSModel(
    Arows,
    Acols,
    Avals,
    nvar,
    b,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    Crows = Crows,
    Ccols = Ccols,
    Cvals = Cvals,
    lcon = lcon,
    ucon = ucon,
    y0 = y0,
    name = name,
  )
end

function LLSModel(
  Arows::AbstractVector{<:Integer},
  Acols::AbstractVector{<:Integer},
  Avals::S,
  nvar::Integer,
  b::S;
  x0::S = fill!(S(undef, nvar), zero(eltype(b))),
  lvar::S = fill!(S(undef, nvar), eltype(b)(-Inf)),
  uvar::S = fill!(S(undef, nvar), eltype(b)(Inf)),
  Crows::AbstractVector{<:Integer} = Int[],
  Ccols::AbstractVector{<:Integer} = Int[],
  Cvals::S = S(undef, 0),
  lcon::S = S(undef, 0),
  ucon::S = S(undef, 0),
  y0::S = fill!(S(undef, length(lcon)), zero(eltype(b))),
  name::String = "generic-LLSModel",
) where {S}
  nequ = length(b)
  ncon = length(lcon)
  if !(ncon == length(ucon) == length(y0))
    error("The length of lcon, ucon and y0 must be the same")
  end
  nnzjF = length(Avals)
  if !(nnzjF == length(Arows) == length(Acols))
    error("The length of Arows, Acols and Avals must be the same")
  end
  nnzj = length(Cvals)
  if !(nnzj == length(Crows) == length(Ccols))
    error("The length of Crows, Ccols and Cvals must be the same")
  end

  meta = NLPModelMeta(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    y0 = y0,
    lin = 1:ncon,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = 0,
    name = name,
  )

  nls_meta = NLSMeta(nequ, nvar, x0 = x0, nnzj = nnzjF, nnzh = 0, lin = 1:nequ)

  return LLSModel(meta, nls_meta, NLSCounters(), Arows, Acols, Avals, b, Crows, Ccols, Cvals)
end

function NLPModels.residual!(nls::LLSModel, x::AbstractVector, Fx::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  coo_prod!(nls.Arows, nls.Acols, nls.Avals, x, Fx)
  Fx .-= nls.b
  return Fx
end

function NLPModels.jac_structure_residual!(
  nls::LLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.nls_meta.nnzj rows
  @lencheck nls.nls_meta.nnzj cols
  rows .= nls.Arows
  cols .= nls.Acols
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls::LLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzj vals
  increment!(nls, :neval_jac_residual)
  vals .= nls.Avals
  return vals
end

function NLPModels.jprod_residual!(
  nls::LLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nvar v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  coo_prod!(nls.Arows, nls.Acols, nls.Avals, v, Jv)
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::LLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck nls.meta.nvar Jtv
  increment!(nls, :neval_jtprod_residual)
  coo_prod!(nls.Acols, nls.Arows, nls.Avals, v, Jtv)
  return Jtv
end

function NLPModels.hess_residual(nls::LLSModel, x::AbstractVector{T}, v::AbstractVector) where {T}
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_hess_residual)
  n = nls.meta.nvar
  return spzeros(T, n, n)
end

function NLPModels.hess_structure_residual!(
  nls::LLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 0 rows
  @lencheck 0 cols
  return rows, cols
end

function NLPModels.hess_coord_residual!(
  nls::LLSModel,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck 0 vals
  increment!(nls, :neval_hess_residual)
  return vals
end

function NLPModels.jth_hess_residual(nls::LLSModel, x::AbstractVector{T}, i::Int) where {T}
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jhess_residual)
  n = nls.meta.nvar
  return spzeros(T, n, n)
end

function NLPModels.hprod_residual!(
  nls::LLSModel,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @lencheck nls.meta.nvar x v Hiv
  increment!(nls, :neval_hprod_residual)
  fill!(Hiv, zero(eltype(x)))
  return Hiv
end

function NLPModels.cons!(nls::LLSModel, x::AbstractVector, c::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon c
  increment!(nls, :neval_cons)
  coo_prod!(nls.Crows, nls.Ccols, nls.Cvals, x, c)
  return c
end

function NLPModels.jac_structure!(
  nls::LLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.meta.nnzj rows cols
  rows .= nls.Crows
  cols .= nls.Ccols
  return rows, cols
end

function NLPModels.jac_coord!(nls::LLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzj vals
  increment!(nls, :neval_jac)
  vals .= nls.Cvals
  return vals
end

function NLPModels.jprod!(nls::LLSModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nls.meta.nvar x v
  @lencheck nls.meta.ncon Jv
  increment!(nls, :neval_jprod)
  coo_prod!(nls.Crows, nls.Ccols, nls.Cvals, v, Jv)
  return Jv
end

function NLPModels.jtprod!(nls::LLSModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.meta.ncon v
  increment!(nls, :neval_jtprod)
  coo_prod!(nls.Ccols, nls.Crows, nls.Cvals, v, Jtv)
  return Jtv
end

function NLPModels.hprod!(
  nls::LLSModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T, S}
  @lencheck nls.meta.nvar x v Hv
  increment!(nls, :neval_hprod)
  Av = fill!(S(undef, nls.nls_meta.nequ), zero(T))
  coo_prod!(nls.Arows, nls.Acols, nls.Avals, v, Av)
  coo_prod!(nls.Acols, nls.Arows, nls.Avals, Av, Hv)
  Hv .*= obj_weight
  return Hv
end

function NLPModels.hprod!(
  nls::LLSModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(eltype(x)),
)
  @lencheck nls.meta.nvar x v Hv
  @lencheck nls.meta.ncon y
  hprod!(nls, x, v, Hv, obj_weight = obj_weight)
end

function NLPModels.ghjvprod!(
  nls::LLSModel,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nls.meta.nvar x g v
  @lencheck nls.meta.ncon gHv
  increment!(nls, :neval_hprod)
  gHv .= zero(T)
  return gHv
end
