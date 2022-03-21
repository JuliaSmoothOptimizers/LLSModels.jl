using Requires

list_krylov_solvers = Dict{Symbol, Symbol}()

@init begin
  @require Krylov = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7" begin
    list_krylov_solvers = Krylov.KRYLOV_SOLVERS

    for (ofun, KS) in list_krylov_solvers
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
        @inline Krylov.$KS(lls::LLSModel, args...; kwargs...) =
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
    @inline Krylov.GpmrSolver(lls::LLSModel, args...; kwargs...) =
      Krylov.GpmrSolver(lls.A, lls.b, args...; kwargs...)

    """
        cg_lanczos(::LLSModel, shifts, args...; kwargs...)

    The Lanczos version of the conjugate gradient method to solve a family of shifted systems for linear least-squares from Krylov.jl.
    """
    function Krylov.cg_lanczos(
      lls::LLSModel,
      shifts::AbstractVector{T},
      args...;
      kwargs...,
    ) where {T <: AbstractFloat}
      unconstrained(lls) || error("The LLSModels has constraints.")
      Krylov.cg_lanczos(lls.A, lls.b, shifts, args...; kwargs...)
    end
    """
        cg_lanczos!(::CgLanczosShiftSolver, ::LLSModel, args...; kwargs...)

    The in-place Lanczos version of the conjugate gradient method to solve a family of shifted systems for linear least-squares from Krylov.jl.
    """
    function Krylov.cg_lanczos!(
      solver::Krylov.CgLanczosShiftSolver,
      lls::LLSModel,
      shifts::AbstractVector{T},
      args...;
      kwargs...,
    ) where {T <: AbstractFloat}
      unconstrained(lls) || error("The LLSModels has constraints.")
      Krylov.cg_lanczos!(solver, lls.A, lls.b, shifts, args...; kwargs...)
    end
    """
        CgLanczosShiftSolver(::LLSModel)

    Wrapper to define the solver structure CgLanczosShiftSolver used in the cg_lanczos! in-place method for linear least-squares from Krylov.jl.
    """
    @inline Krylov.CgLanczosShiftSolver(lls::LLSModel, args...; kwargs...) =
      Krylov.CgLanczosShiftSolver(lls.A, lls.b, args...; kwargs...)
  end
end
