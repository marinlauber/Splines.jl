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
export FEOperator,AbstractFEOperator,integrate!,applyBC!

include("Solver.jl")
export NLsolve,LineSearches,ImplicitAD,GeneralizedAlpha,solve_step!,lsolve!,nlsolve!,global_mass!,residual!,jacobian!


"""
    returns the location of the integration points in physical coordinates
"""
function uv_integration(op::FEOperator)
    B, dB, ddB = bernsteinBasis(op.gauss_rule.nodes, op.mesh.degP[1])
    uv = []
    for iElem = 1:op.mesh.numElem
        #compute the (B-)spline basis functions and derivatives with Bezier extraction
        N_mat = B * op.mesh.C[iElem]'

        #compute the rational spline basis
        curNodes = op.mesh.elemNode[iElem]
        cpts = op.mesh.controlPoints[1, curNodes]
        wgts = op.mesh.weights[curNodes]
        for iGauss = 1:length(op.gauss_rule.nodes)
            #compute the rational basis
            RR = N_mat[iGauss,:].* wgts
            w_sum = sum(RR)
            RR /= w_sum

            # external force at physical point
            phys_pt = RR'*cpts
            push!(uv, phys_pt)
        end
    end
    return uv
end
# reshape to displacements
points(a) = dⁿ(a)
function dⁿ(ga::GeneralizedAlpha)
    return reshape(ga.u[1][1:2ga.op.mesh.numBasis],(ga.op.mesh.numBasis,2))'
end
function vⁿ(ga::GeneralizedAlpha)
    return reshape(ga.u[2][1:2ga.op.mesh.numBasis],(ga.op.mesh.numBasis,2))'
end
export dⁿ,vⁿ,points,uv_integration

end