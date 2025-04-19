module LLSModelsKrylovExt

using Krylov

for (ofun, KS) in Krylov.KRYLOV_SOLVERS
  ofun == :gpmr && continue
  ifun = Symbol(ofun, "!")
  @eval begin
    @doc """
        $(Krylov.$ofun)(::LLSModel, args...; kwargs...)

    Wrapper using the $(Krylov.$ofun) method for linear least-squares from Krylov.jl.
    """
    function Krylov.$(ofun)(lls::LLSModel, args...; kwargs...)
      unconstrained(lls) || error("The LLSModels has constraints.")
      Krylov.$(ofun)(lls.A, lls.b, args...; kwargs...)
    end
    @doc """
        $(Krylov.$ifun)(::$(Krylov.$KS), ::LLSModel, args...; kwargs...)

    Wrapper using the $(Krylov.$ifun) in-place method for linear least-squares from Krylov.jl.
    """
    function Krylov.$(ifun)(solver::Krylov.$KS, lls::LLSModel, args...; kwargs...)
      unconstrained(lls) || error("The LLSModels has constraints.")
      Krylov.$(ifun)(solver, lls.A, lls.b, args...; kwargs...)
    end
    @doc """
        $(Krylov.$KS)(::LLSModel)

    Wrapper to define the solver structure $(Krylov.$KS) used in the $(Krylov.$ifun) in-place method for linear least-squares from Krylov.jl.
    """
    Krylov.$KS(lls::LLSModel, args...; kwargs...) =
      Krylov.$KS(lls.A, lls.b, args...; kwargs...)
  end
end

"""
    gpmr(::LLSModel, args...; kwargs...)

Wrapper using the gpmr method for linear least-squares from Krylov.jl with `B = Aᵀ`.
"""
function Krylov.gpmr(lls::LLSModel, args...; kwargs...)
  unconstrained(lls) || error("The LLSModels has constraints.")
  Krylov.gpmr(lls.A, lls.A', lls.b, args...; kwargs...)
end

"""
    gpmr!(::GpmrSolver, ::LLSModel, args...; kwargs...)

Wrapper using the gpmr! in-place method for linear least-squares from Krylov.jl with `B = Aᵀ`.
"""
function Krylov.gpmr!(solver::Krylov.GpmrSolver, lls::LLSModel, args...; kwargs...)
  unconstrained(lls) || error("The LLSModels has constraints.")
  Krylov.gpmr!(solver, lls.A, lls.A', lls.b, args...; kwargs...)
end

"""
    GpmrSolver(::LLSModel)

Wrapper to define the solver structure GpmrSolver used in the gpmr! in-place method for linear least-squares from Krylov.jl.
"""
Krylov.GpmrSolver(lls::LLSModel, args...; kwargs...) = Krylov.GpmrSolver(lls.A, lls.b, args...; kwargs...)

end
