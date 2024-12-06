using StaticArrays

# abstract type AbstractODEOperator end

# struct ODEOperator <: AbstractODEOperator
#     x :: AbstractArray{T}
#     resid :: AbstractArray{T}
#     jacob :: AbstractArray{T}
#     p :: Dict{String,T}
# end

# struct ODESolver <: AbstractFEOperator end


struct GeneralizedAlpha
    op :: FEOperator
    u ::Union{AbstractVector,Tuple{Vararg{AbstractVector}}}
    αm :: T
    αf :: T
    β :: T
    γ :: T
    op_cache ::Union{AbstractVector,Tuple{Vararg{AbstractVector}}}
    Δt :: Vector{T}
    function GeneralizedAlpha(op::FEOperator, ρ∞::T=0.5) where{T}
        αm = (2.0 - ρ∞)/(ρ∞ + 1.0);
        αf = 1.0/(1.0 + ρ∞)
        γ = 0.5 - αf + αm;
        β = 0.25*(1.0 - αf + αm)^2;
        new(op,(op.d,op.v,op.a),αm,αf,β,γ,(op.d,op.v,op.a),[])
    end
end

"""
    solve_step!(solver::GeneralizedAlpha,force::AbstractArray,Δt::Float64)

steps a (potentially nonlinear) operator forward in time, given an external
force and a time step size.
"""
function solve_step!(solver::GeneralizedAlpha, force, Δt)
    
    # unpack cache
    (dⁿ,vⁿ,aⁿ) = solver.op_cache
              
    # useful
    αf, αm, β, γ = solver.αf, solver.αm, solver.β, solver.γ

    # structural time steps
    tⁿ   = sum(solver.Δt[end])
    tⁿ⁺ᵅ = αf*(tⁿ+Δt) + (1.0-αf)*tⁿ;

    # predictor (initial guess) for the Newton-Raphson scheme
    (dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹) = solver.u
    dⁿ⁺¹ .= dⁿ;
    vⁿ⁺¹ .= vⁿ;
    aⁿ⁺¹ .= aⁿ;

    # Newton-Raphson iterations loop
    r₂ = 1.0; iter = 1;
    while r₂ > 1.0e-6 && iter < 1000

        # compute v_{n+1}, a_{n+1}, ... from "Isogeometric analysis: toward integration of CAD and FEA"
        vⁿ⁺¹ = γ/(β*Δt)*dⁿ⁺¹ - γ/(β*Δt)*dⁿ + (1.0-γ/β)*vⁿ - Δt*(γ/2β-1.0)*aⁿ;
        aⁿ⁺¹ = 1.0/(β*Δt^2)*dⁿ⁺¹ - 1.0/(β*Δt^2)*dⁿ - 1.0/(β*Δt)*vⁿ - (1.0/2β-1.0)*aⁿ;

        # compute d_{n+af}, v_{n+af}, a_{n+am}, ...
        dⁿ⁺ᵅ = αf*dⁿ⁺¹ + (1.0-αf)*dⁿ;
        vⁿ⁺ᵅ = αf*vⁿ⁺¹ + (1.0-αf)*vⁿ;
        aⁿ⁺ᵅ = αm*aⁿ⁺¹ + (1.0-αm)*aⁿ;

        # update the jacobian, the residual and the external force
        integrate!(solver.op, dⁿ⁺ᵅ, force)

        # compute the jacobian and the residuals
        solver.op.jacob .= αm/(β*Δt^2)*solver.op.M + αf*solver.op.jacob
        solver.op.resid = solver.op.stiff*dⁿ⁺ᵅ + solver.op.M*aⁿ⁺ᵅ - solver.op.fext

        # apply BC
        applyBC!(solver.op)

        # check convergence
        r₂ = norm(solver.op.resid);
        if r₂ < 1.0e-6 && break; end

        # newton solve for the displacement increment
        dⁿ⁺¹ -= solver.op.jacob\solver.op.resid; iter += 1
    end

    dⁿ .= dⁿ⁺¹;
    vⁿ .= vⁿ⁺¹;
    aⁿ .= aⁿ⁺¹;

    # copy variables
    solver.u = (dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹)
    solver.op_cache = (dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹)
end