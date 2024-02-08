using NLsolve
using ImplicitAD
using LineSearches
using StaticArrays

# struct GeneralizedAlpha
#     op :: StaticFEOperator
#     u ::Union{AbstractVector,Tuple{Vararg{AbstractVector}}}
#     αm :: Float64
#     αf :: Float64
#     β :: Float64
#     γ :: Float64
#     Δt :: Vector{Float64}
#     function GeneralizedAlpha(opr::StaticFEOperator; ρ∞=1.0)
#         αm = (2.0 - ρ∞)/(ρ∞ + 1.0);
#         αf = 1.0/(1.0 + ρ∞)
#         γ = 0.5 - αf + αm;
#         β = 0.25*(1.0 - αf + αm)^2;
#         new(opr,(zero(opr.resid),zero(opr.resid),zero(opr.resid)),αm,αf,β,γ,[0.0])
#     end
# end

# """
#     solve_step!(solver::GeneralizedAlpha,force::AbstractArray,Δt::Float64)

# steps a (potentially nonlinear) operator forward in time, given an external
# force and a time step size.
# """
# function solve_step!(solver::GeneralizedAlpha, force, Δt)
              
#     # useful
#     αf, αm, β, γ = solver.αf, solver.αm, solver.β, solver.γ

#     # structural time steps
#     tⁿ   = sum(solver.Δt[end])
#     tⁿ⁺ᵅ = αf*(tⁿ+Δt) + (1.0-αf)*tⁿ;

#     # predictor (initial guess) for the Newton-Raphson scheme
#     (dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹) = solver.u
#     dⁿ=copy(dⁿ⁺¹); vⁿ=copy(vⁿ⁺¹); aⁿ=copy(aⁿ⁺¹);

#     # Newton-Raphson iterations loop
#     r₂ = 1.0; iter = 1;
#     while r₂ > 1.0e-6 && iter < 1000

#         # compute v_{n+1}, a_{n+1}, ... from "Isogeometric analysis: toward integration of CAD and FEA"
#         vⁿ⁺¹ = γ/(β*Δt)*dⁿ⁺¹ - γ/(β*Δt)*dⁿ + (1.0-γ/β)*vⁿ - Δt*(γ/2β-1.0)*aⁿ;
#         aⁿ⁺¹ = 1.0/(β*Δt^2)*dⁿ⁺¹ - 1.0/(β*Δt^2)*dⁿ - 1.0/(β*Δt)*vⁿ - (1.0/2β-1.0)*aⁿ;

#         # compute d_{n+af}, v_{n+af}, a_{n+am}, ...
#         dⁿ⁺ᵅ = αf*dⁿ⁺¹ + (1.0-αf)*dⁿ;
#         vⁿ⁺ᵅ = αf*vⁿ⁺¹ + (1.0-αf)*vⁿ;
#         aⁿ⁺ᵅ = αm*aⁿ⁺¹ + (1.0-αm)*aⁿ;

#         # update the jacobian, the residual and the external force
#         integrate!(solver.op, dⁿ⁺ᵅ, force)

#         # compute the jacobian and the residuals
#         solver.op.jacob .= αm/(β*Δt^2)*solver.op.mass .+ αf*solver.op.jacob
#         solver.op.resid .= solver.op.stiff*dⁿ⁺ᵅ + solver.op.mass*aⁿ⁺ᵅ - solver.op.ext

#         # apply BC
#         applyBC!(solver.op)

#         # check convergence
#         r₂ = norm(solver.op.resid);
#         if r₂ < 1.0e-6 && break; end

#         # newton solve for the displacement increment
#         dⁿ⁺¹ -= solver.op.jacob\solver.op.resid; iter += 1
#     end
#     # save variables
#     solver.u[1] .= dⁿ⁺¹
#     solver.u[2] .= vⁿ⁺¹
#     solver.u[3] .= aⁿ⁺¹
# end

function lsolve!(op::AbstractFEOperator, force)

    # update the jacobian, the residual and the external force
    # linearized residuals
    integrate!(op, zero(op.resid), force)

    # compute the residuals
    op.resid .= - op.ext

    # apply BC
    applyBC!(op)

    # solve the system and return
    -op.jacob\op.resid
end

function nlsolve!(op::AbstractFEOperator, x, force)
    
    # unpack pre-allocated storage and the convergence flag
    @unpack resid, jacob = op

    # warp the residual and the jacobian
    f!(resid, x) = residual!(resid, x, force, op)
    j!(jacob, x) = jacobian!(jacob, x, force, op)

    # prepare for solve
    df = NLsolve.OnceDifferentiable(f!, j!, x, resid, jacob)

    # # solve the system
    result = NLsolve.nlsolve(df, x,
                             show_trace=false,
                             linsolve=(x, A, b) -> x .= ImplicitAD.implicit_linear(A, b), # what is that doing?
                             method=:newton,
                             linesearch=LineSearches.BackTracking(maxstep=1e6),
                             ftol=1e-9,
                             iterations=1000)

    # update the state, residual, jacobian, and convergence flag
    x .= result.zero
    op.resid .= df.F
    op.jacob .= df.DF

    return x
end


####



"""
    solve_step!(solver::DynamicFEOpertor,force::AbstractArray,Δt::Float64)

steps a (potentially nonlinear) operator forward in time, given an external
force and a time step size.
"""
function solve_step!(op::DynamicFEOperator, force, Δt)
              
    # useful
    αf, αm, β, γ = op.αf, op.αm, op.β, op.γ

    # structural time steps
    tⁿ   = sum(op.Δt[end])
    tⁿ⁺ᵅ = αf*(tⁿ+Δt) + (1.0-αf)*tⁿ;

    # predictor (initial guess) for the Newton-Raphson scheme
    (dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹) = op.u
    dⁿ=copy(dⁿ⁺¹); vⁿ=copy(vⁿ⁺¹); aⁿ=copy(aⁿ⁺¹);

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
        integrate!(op, dⁿ⁺ᵅ, force)

        # compute the jacobian and the residuals
        op.jacob .= αm/(β*Δt^2)*op.mass .+ αf*op.jacob
        op.resid .= op.stiff*dⁿ⁺ᵅ + op.mass*aⁿ⁺ᵅ - op.ext

        # apply BC
        applyBC!(op)

        # check convergence
        r₂ = norm(op.resid);
        if r₂ < 1.0e-6 && break; end

        # newton solve for the displacement increment
        dⁿ⁺¹ -= op.jacob\op.resid; iter += 1
    end
    # save variables
    op.u[1] .= dⁿ⁺¹
    op.u[2] .= vⁿ⁺¹
    op.u[3] .= aⁿ⁺¹
end


function relaxation!(op::AbstractFEOperator, force, BC=[]; ρ::Function=i->1.0, Δt=0.01, N=10_000)

    mass = zero(op.stiff)
    global_mass!(mass, op.mesh, ρ, op.gauss_rule)
    for i ∈ BC
        mass[:,i] .= 0.
        mass[i,:] .= 0.
        mass[i,i] = 1.0
    end
    v = zero(op.x)
    accel = zero(op.x)
    for it ∈ 1:N
        
        integrate!(op, op.x, force)
        # get force delta
        op.resid .= op.stiff*op.x - op.ext
        # apply BC on the acceleration
        op.resid[BC] .= 0.0

        # compute acceleration
        accel .= op.mass\op.resid
        
        # velocity relaxed
        v .= 0.99*v .+ accel*Δt;
        
        # Update Positions
        delta_u = (v.*Δt)./2;
        delta_u[BC] .= 0.0

        #Total displacements
        op.x .+= delta_u
        norm(delta_u)<√eps() && break
    end
    return op.x
end