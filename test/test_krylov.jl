@testset "test solve LLSModel" begin
  n = 20
  A = spdiagm(-1 => ones(n - 1), 0 => 4 * ones(n), 1 => ones(n - 1))
  b = A * ones(n)
  c = b
  shifts = collect(1.0:10.0)
  nshifts = length(shifts)
  lls = LLSModel(A, b)


for (KS, ofun) in [
  (:LsmrWorkspace     , :lsmr      ),
  (:CgsWorkspace      , :cgs       ),
  (:UsymlqWorkspace   , :usymlq    ),
  (:LnlqWorkspace     , :lnlq      ),
  (:BicgstabWorkspace , :bicgstab  ),
  (:CrlsWorkspace     , :crls      ),
  (:LsqrWorkspace     , :lsqr      ),
  (:MinresWorkspace   , :minres    ),
  (:MinaresWorkspace  , :minares   ),
  (:CgneWorkspace     , :cgne      ),
  (:DqgmresWorkspace  , :dqgmres   ),
  (:SymmlqWorkspace   , :symmlq    ),
  (:TrimrWorkspace    , :trimr     ),
  (:UsymqrWorkspace   , :usymqr    ),
  (:BilqrWorkspace    , :bilqr     ),
  (:CrWorkspace       , :cr        ),
  (:CarWorkspace      , :car       ),
  (:CraigmrWorkspace  , :craigmr   ),
  (:TricgWorkspace    , :tricg     ),
  (:CraigWorkspace    , :craig     ),
  (:DiomWorkspace     , :diom      ),
  (:LslqWorkspace     , :lslq      ),
  (:TrilqrWorkspace   , :trilqr    ),
  (:CrmrWorkspace     , :crmr      ),
  (:CgWorkspace       , :cg        ),
  (:CglsWorkspace     , :cgls      ),
  (:CgLanczosWorkspace, :cg_lanczos),
  (:BilqWorkspace     , :bilq      ),
  (:MinresQlpWorkspace, :minres_qlp),
  (:QmrWorkspace      , :qmr       ),
  (:GmresWorkspace    , :gmres     ),
  (:FgmresWorkspace   , :fgmres    ),
  (:FomWorkspace      , :fom       ),
  (:GpmrWorkspace     , :gpmr      ),
  (:CgLanczosShiftWorkspace  , :cg_lanczos_shift  ),
  (:CglsLanczosShiftWorkspace, :cgls_lanczos_shift),
]
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
    elseif ofun in [:cg_lanczos_shift, :cgls_lanczos_shift]
      (x, stats) = eval(ofun)(lls, shifts)
      @test stats.solved
    else
      (x, stats) = eval(ofun)(lls)
      @test stats.solved
    end

    if KS in [:CgLanczosShiftWorkspace, :CglsLanczosShiftWorkspace]
      workspace = eval(KS)(lls, nshifts)
    else
      workspace = eval(KS)(lls)
    end

    ifun = Symbol(ofun, "!")
    if ifun in [:bilqr!, :gpmr!, :tricg!, :trilqr!, :trimr!, :usymlq!, :usymqr!]
      eval(ifun)(workspace, lls, c)
    elseif ifun in [:cg_lanczos_shift!, :cgls_lanczos_shift!]
      eval(ifun)(workspace, lls, shifts)
    else
      eval(ifun)(workspace, lls)
    end
    @test Krylov.issolved(workspace)
  end
end
