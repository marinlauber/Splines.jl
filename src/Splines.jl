module Splines

using UnPack
using Plots

export Plots
export @unpack

include("bernstein.jl")
export norm

include("NURBS.jl")
# export NURBS,nrbmak,findspan,nrbline

include("Mesh.jl")
export Mesh1D,getSol

include("Boundary.jl")
export Boundary1D

include("Quadrature.jl")

include("Operator.jl")
export AbstractFEOperator,StaticFEOperator,DynamicFEOperator,integrate!,applyBC!

include("Solver.jl")
export NLsolve,LineSearches,ImplicitAD,GeneralizedAlpha,Newmark
export lsolve!,nlsolve!,solve_step!,global_mass!,residual!,jacobian!

include("utils.jl")
export dⁿ,vⁿ,points,uv_integration

end