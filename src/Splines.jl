module Splines

using Plots
export Plots

include("bernstein.jl")
export norm

include("NURBS.jl")
# export NURBS,nrbmak,findspan,nrbline

include("Mesh.jl")
export Mesh1D,getSol

include("Boundary.jl")
export Boundary1D

include("Quadrature.jl")
# export GaussQuad,genGaussLegendre

include("EulerBeam.jl")
export @unpack,Problem1D,static_residuals!,static_jacobian!,dynamic_residuals!,dynamic_jacobian!

include("Solver.jl")
export NLsolve,LineSearches,ImplicitAD,static_lsolve!,static_nlsolve!

"""
    make a sparse copy of a mtrix
"""
spcopy(a) = sparse(copy(a))
spzero(a) = sparse(zero(a))
export spcopy, spzero

end