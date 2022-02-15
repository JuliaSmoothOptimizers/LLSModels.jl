@test LLSModels.list_krylov_solvers == Dict()

using Krylov

@testset "test solve LLSModel" begin
  n = 20
  A = spdiagm(-1 => ones(n-1), 0 => 4*ones(n), 1 => ones(n-1))
  b = A * ones(n)
  lls = LLSModel(A, b)

  (x, stats) = cgls(lls)
  solver = CglsSolver(lls)
  cgls!(solver, lls)
  @test stats.solved

  for (ofun, KS) in LLSModels.list_krylov_solvers
    if ofun in [:craig, :craigmr, :lnlq]
      (x, y, stats) = eval(ofun)(lls)
      @test stats.solved
    else
      (x, stats) = eval(ofun)(lls)
      @test stats.solved
    end
    solver = if ofun in [:fom, :gmres]
      eval(KS)(lls)
    else
      eval(KS)(lls)
    end
    ifun = Symbol(ofun, "!")
    eval(ifun)(solver, lls)
    @test issolved(solver)
  end
end
