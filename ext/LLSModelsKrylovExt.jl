module LLSModelsKrylovExt

using Krylov, LLSModels, NLPModels

for (KS, ofun) in [
  (:LsmrWorkspace, :lsmr),
  (:CgsWorkspace, :cgs),
  (:UsymlqWorkspace, :usymlq),
  (:LnlqWorkspace, :lnlq),
  (:BicgstabWorkspace, :bicgstab),
  (:CrlsWorkspace, :crls),
  (:LsqrWorkspace, :lsqr),
  (:MinresWorkspace, :minres),
  (:MinaresWorkspace, :minares),
  (:CgneWorkspace, :cgne),
  (:DqgmresWorkspace, :dqgmres),
  (:SymmlqWorkspace, :symmlq),
  (:TrimrWorkspace, :trimr),
  (:UsymqrWorkspace, :usymqr),
  (:BilqrWorkspace, :bilqr),
  (:CrWorkspace, :cr),
  (:CarWorkspace, :car),
  (:CraigmrWorkspace, :craigmr),
  (:TricgWorkspace, :tricg),
  (:CraigWorkspace, :craig),
  (:DiomWorkspace, :diom),
  (:LslqWorkspace, :lslq),
  (:TrilqrWorkspace, :trilqr),
  (:CrmrWorkspace, :crmr),
  (:CgWorkspace, :cg),
  (:CglsWorkspace, :cgls),
  (:CgLanczosWorkspace, :cg_lanczos),
  (:BilqWorkspace, :bilq),
  (:MinresQlpWorkspace, :minres_qlp),
  (:QmrWorkspace, :qmr),
  (:GmresWorkspace, :gmres),
  (:FgmresWorkspace, :fgmres),
  (:FomWorkspace, :fom),
  (:CgLanczosShiftWorkspace, :cg_lanczos_shift),
  (:CglsLanczosShiftWorkspace, :cgls_lanczos_shift),
]
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
    Krylov.$KS(lls::LLSModel, args...; kwargs...) = Krylov.$KS(lls.A, lls.b, args...; kwargs...)
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
    gpmr!(::GpmrWorkspace, ::LLSModel, args...; kwargs...)

Wrapper using the gpmr! in-place method for linear least-squares from Krylov.jl with `B = Aᵀ`.
"""
function Krylov.gpmr!(solver::Krylov.GpmrWorkspace, lls::LLSModel, args...; kwargs...)
  unconstrained(lls) || error("The LLSModels has constraints.")
  Krylov.gpmr!(solver, lls.A, lls.A', lls.b, args...; kwargs...)
end

"""
    GpmrWorkspace(::LLSModel)

Wrapper to define the solver structure GpmrWorkspace used in the gpmr! in-place method for linear least-squares from Krylov.jl.
"""
Krylov.GpmrWorkspace(lls::LLSModel, args...; kwargs...) =
  Krylov.GpmrWorkspace(lls.A, lls.b, args...; kwargs...)

end
