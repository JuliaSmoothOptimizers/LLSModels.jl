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

  A::SparseMatrixCOO{T, Int}
  b::S
  C::SparseMatrixCOO{T, Int}
end

NLPModels.show_header(io::IO, nls::LLSModel) = println(io, "LLSModel - Linear least-squares model")

function LLSModel(
  A::AbstractMatrix,
  b::S;
  x0::S = fill!(S(undef, size(A, 2)), zero(eltype(b))),
  lvar::S = fill!(S(undef, size(A, 2)), eltype(b)(-Inf)),
  uvar::S = fill!(S(undef, size(A, 2)), eltype(b)(Inf)),
  lcon::S = S(undef, 0),
  ucon::S = S(undef, 0),
  C::AbstractMatrix = SparseMatrixCOO(length(lcon), length(lvar), Int[], Int[], similar(b, 0, 0)),
  y0::S = fill!(S(undef, size(C, 1)), zero(eltype(b))),
  name::String = "generic-LLSModel",
) where {S}
  ncon, nvar = size(A)
  m, n = size(C)
  Arows, Acols, Avals = if A isa AbstractSparseMatrix
    findnz(A)
  else
    I = ((i, j) for i = 1:ncon, j = 1:nvar)
    getindex.(I, 1)[:], getindex.(I, 2)[:], A[:]
  end
  Crows, Ccols, Cvals = if C isa AbstractSparseMatrix
    findnz(C)
  else
    I = ((i, j) for i = 1:m, j = 1:n)
    getindex.(I, 1)[:], getindex.(I, 2)[:], C[:]
  end
  LLSModel(
    SparseMatrixCOO(ncon, nvar, Arows, Acols, Avals),
    nvar,
    b,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    C = SparseMatrixCOO(m, n, Crows, Ccols, Cvals),
    lcon = lcon,
    ucon = ucon,
    y0 = y0,
    name = name,
  )
end

function LLSModel(
  A::SparseMatrixCOO{T, <:Integer},
  nvar::Integer,
  b::S;
  x0::S = fill!(S(undef, nvar), zero(eltype(b))),
  lvar::S = fill!(S(undef, nvar), eltype(b)(-Inf)),
  uvar::S = fill!(S(undef, nvar), eltype(b)(Inf)),
  lcon::S = S(undef, 0),
  ucon::S = S(undef, 0),
  C::SparseMatrixCOO{T, <:Integer} = SparseMatrixCOO(length(lcon), nvar, Int[], Int[], S(undef, 0)),
  y0::S = fill!(S(undef, length(lcon)), zero(eltype(b))),
  name::String = "generic-LLSModel",
) where {T, S}
  nequ = length(b)
  ncon = length(lcon)

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
    nnzj = nnz(C),
    nnzh = 0,
    name = name,
  )

  nls_meta = NLSMeta(nequ, nvar, x0 = x0, nnzj = nnz(A), nnzh = 0, lin = 1:nequ)

  return LLSModel(meta, nls_meta, NLSCounters(), A, b, C)
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
) where {T, S}

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

  return LLSModel(
    SparseMatrixCOO(nequ, nvar, Arows, Acols, Avals),
    nvar,
    b;
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    y0 = y0,
    C = SparseMatrixCOO(ncon, nvar, Crows, Ccols, Cvals),
    lcon = lcon,
    ucon = ucon,
    name = name,
  )
end

function NLPModels.residual!(nls::LLSModel, x::AbstractVector, Fx::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  mul!(Fx, nls.A, x)
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
  rows .= nls.A.rows
  cols .= nls.A.cols
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls::LLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzj vals
  increment!(nls, :neval_jac_residual)
  vals .= nls.A.vals
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
  mul!(Jv, nls.A, v)
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
  mul!(Jtv, transpose(nls.A), v)
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
  mul!(c, nls.C, x)
  return c
end

function NLPModels.jac_structure!(
  nls::LLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.meta.nnzj rows cols
  rows .= nls.C.rows
  cols .= nls.C.cols
  return rows, cols
end

function NLPModels.jac_coord!(nls::LLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzj vals
  increment!(nls, :neval_jac)
  vals .= nls.C.vals
  return vals
end

function NLPModels.jprod!(nls::LLSModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nls.meta.nvar x v
  @lencheck nls.meta.ncon Jv
  increment!(nls, :neval_jprod)
  mul!(Jv, nls.C, v)
  return Jv
end

function NLPModels.jtprod!(nls::LLSModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.meta.ncon v
  increment!(nls, :neval_jtprod)
  mul!(Jtv, transpose(nls.C), v)
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
  mul!(Av, nls.A, v)
  mul!(Hv, transpose(nls.A), Av)
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
