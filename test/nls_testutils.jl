@testset "Checking TestUtils tests on problem LLS" begin
  lls_from_T(::Type{T} = Float64) where {T} = LLSModel(
    T[1 -1; 1 1; 0 1],
    T[0; 2; 2],
    x0 = zeros(T, 2),
    C = T[1 1],
    lcon = T[0],
    ucon = T[Inf],
    name = "lls_LLSModel",
  )
  lls = lls_from_T()
  nls_man = LLS()

  show(IOBuffer(), lls)

  @testset "Check Consistency" begin
    consistent_nlss([lls; nls_man], linear_api = true)
  end
  @testset "Check dimensions" begin
    check_nls_dimensions(lls)
    check_nlp_dimensions(lls, exclude = [hess, hess_coord, jth_hess, jth_hess_coord, jth_hprod], linear_api = true)
  end
  @testset "Multiple precision support" begin
    multiple_precision_nls(lls_from_T, linear_api = true)
  end
  @testset "Check view subarray" begin
    view_subarray_nls(lls)
  end
end
