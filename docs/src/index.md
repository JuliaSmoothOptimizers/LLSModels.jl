# [LLSModels.jl documentation](@id Home)

This package provides a linear least squares model implementing the [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API.

## Install

Install LLSModels.jl with the following command.
```julia
pkg> add LLSModels
```

## Usage

This package defines [`LLSModel`](@ref).

```@docs
LLSModel
```

We can define a linear least squares by passing the matrices that define the problem
```math
\begin{aligned}
\min \quad & \tfrac{1}{2}\|Ax - b\|^2 \\
& c_L  \leq Cx \leq c_U \\
& \ell \leq  x \leq u.
\end{aligned}
```
```@example nls
using LinearAlgebra, LLSModels, NLPModels # hide
A = rand(10, 3)
b = rand(10)
C = rand(2, 3)
nls = LLSModel(A, b, C=C, lcon=zeros(2), ucon=zeros(2), lvar=-ones(3), uvar=ones(3))
```

## License

This content is released under the [MPL2.0](https://www.mozilla.org/en-US/MPL/2.0/) License.

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/LLSModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.

## Contents

```@contents
```
