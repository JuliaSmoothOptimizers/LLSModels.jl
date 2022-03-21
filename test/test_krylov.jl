@test LLSModels.list_krylov_solvers == Dict()

using Krylov

@testset "test solve LLSModel" begin
  n = 20
  A = spdiagm(-1 => ones(n-1), 0 => 4*ones(n), 1 => ones(n-1))
  b = A * ones(n)
  c = b
  lls = LLSModel(A, b)

  (x, stats) = cgls(lls)
  solver = CglsSolver(lls)
  cgls!(solver, lls)
  @test stats.solved

  shifts = collect(1.:10.)
  nshifts = length(shifts)
  lls = LLSModel(A, b)
  (x, stats) = cg_lanczos(lls, shifts)
  @test stats.solved
  solver = CgLanczosShiftSolver(lls, nshifts)
  cg_lanczos!(solver, lls, shifts)
  @test stats.solved

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
    else
      (x, stats) = eval(ofun)(lls)
      @test stats.solved
    end
    solver = eval(KS)(lls)
    ifun = Symbol(ofun, "!")
    if ofun in [:bilqr, :gpmr, :tricg, :trilqr, :trimr, :usymlq, :usymqr]
      eval(ifun)(solver, lls, c)
    else
      eval(ifun)(solver, lls)
    end
    @test issolved(solver)
  end
end
