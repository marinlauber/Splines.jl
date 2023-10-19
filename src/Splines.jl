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
export @unpack,EulerBeam,static_residuals!,static_jacobian!,dynamic_residuals!,dynamic_jacobian!
export global_mass!,dynamic_update!

include("Solver.jl")
export NLsolve,LineSearches,ImplicitAD,static_lsolve!,static_nlsolve!

"""
    make a sparse copy of a mtrix
"""
spcopy(a) = sparse(copy(a))
spzero(a) = sparse(zero(a))
export spcopy, spzero

function uv_integration(p::EulerBeam)
    B, dB, ddB = Splines.bernsteinBasis(p.gauss_rule.nodes, p.mesh.degP[1])
    uv = []
    for iElem = 1:p.mesh.numElem
        #compute the (B-)spline basis functions and derivatives with Bezier extraction
        N_mat = B * p.mesh.C[iElem]'

        #compute the rational spline basis
        curNodes = p.mesh.elemNode[iElem]
        cpts = p.mesh.controlPoints[1, curNodes]
        wgts = p.mesh.weights[curNodes]
        for iGauss = 1:length(p.gauss_rule.nodes)
            #compute the rational basis
            RR = N_mat[iGauss,:].* wgts
            w_sum = sum(RR)
            RR /= w_sum

            # external force at physical point
            phys_pt = RR'*cpts
            # println((iElem-1)*p.mesh.numElem+iGauss) #index
            push!(uv, phys_pt)
        end
    end
    return uv
end

"""
    step in time
"""
function step(jacob::Matrix{T},stiff::Matrix{T},M::Matrix{T},
              resid::Vector{T},fext::Vector{T},loading::Vector{T},
              dⁿ::Vector{T},vⁿ::Vector{T},aⁿ::Vector{T},
              tⁿ::T,tⁿ⁺¹::T,αm::T,αf::T,β::T,γ::T,p::EulerBeam) where{T}

    # structural time steps
    Δt = tⁿ⁺¹ - tⁿ;
    tⁿ⁺ᵅ = αf*tⁿ⁺¹ + (1.0-αf)*tⁿ;
    
    # predictor (initial guess) for the Newton-Raphson scheme
    # d_{n+1}
    dⁿ⁺¹ = copy(dⁿ); r₂ = 1.0; iter = 1;
    vⁿ⁺¹ = copy(vⁿ);
    aⁿ⁺¹ = copy(aⁿ);

    # Newton-Raphson iterations loop
    while r₂ > 1.0e-6 && iter < 1000
        # compute v_{n+1}, a_{n+1}, ... from "Isogeometric analysis: toward integration of CAD and FEA"
        vⁿ⁺¹ = γ/(β*Δt)*dⁿ⁺¹ - γ/(β*Δt)*dⁿ + (1.0-γ/β)*vⁿ - Δt*(γ/2β-1.0)*aⁿ;
        aⁿ⁺¹ = 1.0/(β*Δt^2)*dⁿ⁺¹ - 1.0/(β*Δt^2)*dⁿ - 1.0/(β*Δt)*vⁿ - (1.0/2β-1.0)*aⁿ;

        # compute d_{n+af}, v_{n+af}, a_{n+am}, ...
        dⁿ⁺ᵅ = αf*dⁿ⁺¹ + (1.0-αf)*dⁿ;
        vⁿ⁺ᵅ = αf*vⁿ⁺¹ + (1.0-αf)*vⁿ;
        aⁿ⁺ᵅ = αm*aⁿ⁺¹ + (1.0-αm)*aⁿ;
    
        # update stiffness and jacobian, linearised here
        Splines.update_global!(stiff, jacob, dⁿ⁺ᵅ, p.mesh, p.gauss_rule, p)
    
        # update rhs vector
        Splines.update_external!(fext, p.mesh, p.f, p.gauss_rule)
        fext .+= loading;

        # apply BCs
        jacob .= αm/(β*Δt^2)*M + αf*jacob
        resid .= stiff*dⁿ⁺ᵅ + M*aⁿ⁺ᵅ - fext
        Splines.applyBCGlobal!(stiff, jacob, resid, p.mesh, 
                               p.Dirichlet_BC, p.Neumann_BC,
                               p.gauss_rule)

        # check convergence
        r₂ = norm(resid);
        if r₂ < 1.0e-6 && break; end

        # newton solve for the displacement increment
        dⁿ⁺¹ -= jacob\resid; iter += 1
    end
    
    # copy variables ()_{n} <-- ()_{n+1}
    return dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹
end

function step2(jacob::Matrix{T},stiff::Matrix{T},M::Matrix{T},
               resid::Vector{T},fext::Vector{T},loading::Matrix{T},
               dⁿ::Vector{T},vⁿ::Vector{T},aⁿ::Vector{T},
               tⁿ::T,tⁿ⁺¹::T,αm::T,αf::T,β::T,γ::T,p::EulerBeam) where{T}

    # structural time steps
    Δt = tⁿ⁺¹ - tⁿ;
    tⁿ⁺ᵅ = αf*tⁿ⁺¹ + (1.0-αf)*tⁿ;

    # predictor (initial guess) for the Newton-Raphson scheme
    # d_{n+1}
    dⁿ⁺¹ = copy(dⁿ); r₂ = 1.0; iter = 1;
    vⁿ⁺¹ = copy(vⁿ);
    aⁿ⁺¹ = copy(aⁿ);

    # Newton-Raphson iterations loop
    while r₂ > 1.0e-6 && iter < 1000
        # compute v_{n+1}, a_{n+1}, ... from "Isogeometric analysis: toward integration of CAD and FEA"
        vⁿ⁺¹ = γ/(β*Δt)*dⁿ⁺¹ - γ/(β*Δt)*dⁿ + (1.0-γ/β)*vⁿ - Δt*(γ/2β-1.0)*aⁿ;
        aⁿ⁺¹ = 1.0/(β*Δt^2)*dⁿ⁺¹ - 1.0/(β*Δt^2)*dⁿ - 1.0/(β*Δt)*vⁿ - (1.0/2β-1.0)*aⁿ;

        # compute d_{n+af}, v_{n+af}, a_{n+am}, ...
        dⁿ⁺ᵅ = αf*dⁿ⁺¹ + (1.0-αf)*dⁿ;
        vⁿ⁺ᵅ = αf*vⁿ⁺¹ + (1.0-αf)*vⁿ;
        aⁿ⁺ᵅ = αm*aⁿ⁺¹ + (1.0-αm)*aⁿ;

        # update stiffness and jacobian, linearised here
        Splines.update_global!(stiff, jacob, dⁿ⁺ᵅ, p.mesh, p.gauss_rule, p)

        # update rhs vector
        Splines.update_external!(fext, p.mesh, loading, p.gauss_rule)

        # apply BCs
        jacob .= αm/(β*Δt^2)*M + αf*jacob
        resid .= stiff*dⁿ⁺ᵅ + M*aⁿ⁺ᵅ - fext
        Splines.applyBCGlobal!(stiff, jacob, resid, p.mesh, 
                               p.Dirichlet_BC, p.Neumann_BC,
                               p.gauss_rule)

        # check convergence
        r₂ = norm(resid);
        if r₂ < 1.0e-6 && break; end

        # newton solve for the displacement increment
        dⁿ⁺¹ -= jacob\resid; iter += 1
    end

    # copy variables ()_{n} <-- ()_{n+1}
    return dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹
end

end