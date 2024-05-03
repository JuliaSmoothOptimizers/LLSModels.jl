@testset "Checking TestUtils tests on problem LLS" begin
  lls_from_T(::Type{T} = Float64) where {T <: Number} = lls_from_T(Vector{Float64}, Matrix{Float64})
  lls_from_T(::Type{S}, ::Type{Mat}) where {Mat, S} = LLSModel(
    Mat([1 -1; 1 1; 0 1]),
    S([0; 2; 2]),
    x0 = fill!(S(undef, 2), 0),
    C = Mat([1 1]),
    lcon = fill!(S(undef, 1), 0),
    ucon = fill!(S(undef, 1), Inf),
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
    check_nlp_dimensions(
      lls,
      exclude = [hess, hess_coord, jth_hess, jth_hess_coord, jth_hprod],
      linear_api = true,
    )
  end
  @testset "Multiple precision support" begin
    multiple_precision_nls(lls_from_T, linear_api = true)
  end
  if CUDA.functional()
    @testset "GPU Multiple precision support" begin
      CUDA.allowscalar() do
        multiple_precision_nls_array(lls_from_T, CuArray, linear_api = true)
      end
    end
  end
  @testset "Check view subarray" begin
    view_subarray_nls(lls)
  end
end
