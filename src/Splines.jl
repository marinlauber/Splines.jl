module Splines

using Plots
export Plots

include("bernstein.jl")
export norm

include("NURBS.jl")
# export NURBS,nrbmak,findspan,nrbline

include("Mesh.jl")
export Mesh1D

include("Boundary.jl")
export Boundary1D

include("Quadrature.jl")
# export GaussQuad,genGaussLegendre

include("EulerBeam.jl")
export @unpack,Problem1D,assemble_mass,assemble_rhs,assemble_stiff,assemble,applyBCDirichlet,applyBCNeumann,getSol

include("Solver.jl")
export NLsolve,LineSearches,ImplicitAD,static_residuals!,static_jacobian!,static_lsolve!,static_nlsolve!

end