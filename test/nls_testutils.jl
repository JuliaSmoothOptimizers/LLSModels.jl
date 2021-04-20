@testset "Checking TestUtils tests on problem LLS" begin
  lls = LLSModel(
    [1.0 -1; 1 1; 0 1],
    [0.0; 2; 2],
    x0 = zeros(2),
    C = [1.0 1],
    lcon = [0.0],
    ucon = [Inf],
    name = "lls_LLSModel",
  )
  nls_man = LLS()

  show(IOBuffer(), lls)

  @testset "Check Consistency" begin
    consistent_nlss([lls; nls_man])
  end
  @testset "Check dimensions" begin
    check_nls_dimensions(lls)
    check_nlp_dimensions(lls, exclude = [hess, hess_coord, jth_hess, jth_hess_coord, jth_hprod])
  end
  @testset "Multiple precision support" begin
    multiple_precision_nls(lls)
  end
  @testset "Check view subarray" begin
    view_subarray_nls(lls)
  end
end
