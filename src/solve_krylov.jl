using Requires

list_krylov_solvers = Dict{Symbol,Symbol}()

@init begin
  @require Krylov = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7" begin
    list_krylov_solvers = Krylov.KRYLOV_SOLVERS
    for k in [:bilqr, :gpmr, :tricg, :trilqr, :trimr, :usymlq, :usymqr]
      delete!(list_krylov_solvers, k)
    end

    for (ofun, KS) in list_krylov_solvers
      ifun = Symbol(ofun, "!")
      @eval begin
        @doc """
            $(Krylov.$ofun)(::LLSModel; kwargs...)

        Wrapper using the $(Krylov.$ofun) method for linear least-squares from Krylov.jl.
        """
        function Krylov.$(ofun)(lls::LLSModel; kwargs...)
          unconstrained(lls) ||  error("The LLSModels has constraints.")
          Krylov.$(ofun)(lls.A, lls.b; kwargs...)
        end
        @doc """
            $(Krylov.$ifun)(::$(Krylov.$KS), ::LLSModel; kwargs...)

        Wrapper using the $(Krylov.$ifun) in-place method for linear least-squares from Krylov.jl.
        """
        function Krylov.$(ifun)(solver :: Krylov.$KS, lls::LLSModel; kwargs...)
          unconstrained(lls) ||  error("The LLSModels has constraints.")
          Krylov.$(ifun)(solver, lls.A, lls.b; kwargs...)
        end
        @doc """
            $(Krylov.$KS)(::LLSModel)

        Wrapper to define the solver structure $(Krylov.$KS) used in the $(Krylov.$ifun) in-place method for linear least-squares from Krylov.jl.
        """
        @inline Krylov.$KS(lls::LLSModel, args...; kwargs...) = Krylov.$KS(lls.A, lls.b, args...; kwargs...)
      end
    end
  end
end
