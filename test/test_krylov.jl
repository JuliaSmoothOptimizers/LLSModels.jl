@test LLSModels.list_krylov_solvers == Dict()

using Krylov

@testset "test solve LLSModel" begin
  n = 20
  A = spdiagm(-1 => ones(n - 1), 0 => 4 * ones(n), 1 => ones(n - 1))
  b = A * ones(n)
  c = b
  shifts = collect(1.0:10.0)
  nshifts = length(shifts)
  lls = LLSModel(A, b)

  for (ofun, KS) in LLSModels.list_krylov_solvers
    if ofun in [:craig, :craigmr, :lnlq]
      (x, y, stats) = eval(ofun)(lls)
      @test stats.solved
    elseif ofun in [:bilqr, :trilqr]
      (x, y, stats) = eval(ofun)(lls, c)
      @test stats.solved_primal && stats.solved_dual
    elseif ofun in [:gpmr, :tricg, :trimr]
      (x, y, stats) = eval(ofun)(lls, c)
      @test stats.solved
    elseif ofun in [:usymlq, :usymqr]
      (x, stats) = eval(ofun)(lls, c)
      @test stats.solved
    elseif ofun == :cg_lanczos_shift
      (x, stats) = eval(ofun)(lls, shifts)
      @test stats.solved
    else
      (x, stats) = eval(ofun)(lls)
      @test stats.solved
    end

    if KS == :CgLanczosShiftSolver
      solver = eval(KS)(lls, nshifts)
    else
      solver = eval(KS)(lls)
    end

    ifun = Symbol(ofun, "!")
    if ifun in [:bilqr!, :gpmr!, :tricg!, :trilqr!, :trimr!, :usymlq!, :usymqr!]
      eval(ifun)(solver, lls, c)
    elseif ifun == :cg_lanczos_shift!
      eval(ifun)(solver, lls, shifts)
    else
      eval(ifun)(solver, lls)
    end
    @test issolved(solver)
  end
end
