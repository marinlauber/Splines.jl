using NLsolve
using ImplicitAD
using LineSearches
using StaticArrays

"""
    lsolve(op::AbstractFEOperator)

Solves the linearized system of equations for the operator `op` and returns the
displacement increment.
"""
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

"""
    nlsolve!(op::AbstractFEOperator)

Solves the nonlinear system of equations for the operator `op` and returns the
displacement increment.
"""
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

"""
    solve_step!(solver::DynamicFEOpertor,force::AbstractArray,Δt::Float64)

Integrate a (potentially nonlinear) operator forward in time, given an external
force and a time step size.
"""
function solve_step!(op::DynamicFEOperator{GeneralizedAlpha}, force, Δt; tol=1.0e-6, max_iter=1000)
              
    # useful
    αf, αm, β, γ = op.αf, op.αm, op.β, op.γ

    # predictor (initial guess) for the Newton-Raphson scheme
    (dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹) = op.u
    dⁿ=copy(dⁿ⁺¹); vⁿ=copy(vⁿ⁺¹); aⁿ=copy(aⁿ⁺¹);

    # Newton-Raphson iterations loop
    r₂ = 1.0; iter = 1;
    while r₂ > tol && iter < max_iter

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
        if r₂ < tol && break; end

        # newton solve for the displacement increment
        dⁿ⁺¹ -= op.jacob\op.resid; iter += 1
    end
    # save variables
    op.u[1] .= dⁿ⁺¹
    op.u[2] .= vⁿ⁺¹
    op.u[3] .= aⁿ⁺¹
end
# @benchmark solve_step!(operator, f_ext, Δt)
# BenchmarkTools.Trial: 88 samples with 1 evaluation.
#  Range (min … max):  50.358 ms … 87.329 ms  ┊ GC (min … max): 6.94% … 8.70%
#  Time  (median):     56.670 ms              ┊ GC (median):    7.67%
#  Time  (mean ± σ):   56.888 ms ±  3.881 ms  ┊ GC (mean ± σ):  7.84% ± 0.99%

#                              ▇▁▂██ ▁▁                          
#   ▅▅▁▁▃▁▁▁▁▁▁▅▁▃▁▁▁▁▁▁▁▁▅▁▁█▆█████▆███▆▁▃▅▁▃▁▅▃▁▁▁▁▁▁▁▁▁▃▃▁▁▃ ▁
#   50.4 ms         Histogram: frequency by time        62.5 ms <

#  Memory estimate: 181.75 MiB, allocs estimate: 1520493.
function solve_step!(op::DynamicFEOperator{Newmark}, force, Δt; tol=1.0e-6, max_iter=1000)
              
    # useful
    β, γ = op.β, op.γ

    # predictor (initial guess) for the Newton-Raphson scheme
    (dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹) = op.u

    # predictor step
    dⁿ⁺¹ += Δt*vⁿ⁺¹ + Δt^2/2*(1-2*β)*aⁿ⁺¹;
    vⁿ⁺¹ += Δt*(1-γ)*aⁿ⁺¹;
    aⁿ⁺¹ .= 0.0;

    # Newton-Raphson iterations loop
    r₂ = 1.0; iter = 1;
    while r₂ > tol && iter < max_iter

        # update the jacobian, the residual and the external force
        integrate!(op, dⁿ⁺¹, force)

        # compute the jacobian and the residuals
        op.jacob .= 1.0/(β*Δt^2)*op.mass .+ op.jacob
        op.resid .= op.stiff*dⁿ⁺¹ + op.mass*aⁿ⁺¹ - op.ext

        # apply BC
        applyBC!(op)

        # check convergence
        r₂ = norm(op.resid);
        if r₂ < tol && break; end

        # newton solve for the displacement increment
        Δd = -op.jacob\op.resid; iter += 1
        dⁿ⁺¹ += Δd
        vⁿ⁺¹ += γ/(β*Δt)*Δd
        aⁿ⁺¹ += 1.0/(β*Δt^2)*Δd
    end
    # save variables
    op.u[1] .= dⁿ⁺¹
    op.u[2] .= vⁿ⁺¹
    op.u[3] .= aⁿ⁺¹
end
#@TODO: remove if not used
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