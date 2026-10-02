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
    op.resid .= -op.ext

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
LinearAlgebra.norm(op::AbstractFEOperator) = norm(op.resid[1:2op.mesh.numBasis])
function nlsolve!(op::AbstractFEOperator, x, force; tol=1.0e-6, max_iter=1000)
    # Newton-Raphson iterations loop
    r₂ = 1.0; r_prev = Inf; iter = 1;
    while r₂ > tol && iter < max_iter
        # update the jacobian, the residual and the external force
        integrate!(op, x, force)

        # compute residuals
        op.resid .=  op.stiff*x - op.ext

        # apply BC (the iterate is what puts the Neumann reaction into the residual)
        applyBC!(op, x)

        # check convergence
        r₂ = norm(op)
        # converged, or the iteration has stopped making progress. The second test is not
        # a nicety: `op.resid` is formed from `op.stiff`, which `integrate!` zeroes and
        # rebuilds each pass, so it never carries the Neumann Lagrange coupling that
        # `applyBC!` puts into `op.jacob`. The residual therefore settles on a floor the
        # Newton direction cannot reach, and every further step adds an increment that is
        # invisible to the residual -- it only inflates the multipliers. Without this the
        # loop runs all `max_iter` steps where two do the work.
        (r₂ < tol || r₂ > 0.99r_prev) && break
        r_prev = r₂

        # newton solve for the displacement increment
        x .-= op.jacob\op.resid; iter += 1
    end
end

"""
    solve_step!(solver::DynamicFEOpertor,force::AbstractArray,Δt::Float64)

Integrate a (potentially nonlinear) operator forward in time, given an external
force and a time step size.
"""
function solve_step!(op::DynamicFEOperator{GeneralizedAlpha}, force, Δt; tol=1e-4, max_iter=1e3)

    # useful
    αf, αm, β, γ = op.αf, op.αm, op.β, op.γ

    # predictor (initial guess) for the Newton-Raphson scheme
    (dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹) = op.u
    dⁿ=copy(dⁿ⁺¹); vⁿ=copy(vⁿ⁺¹); aⁿ=copy(aⁿ⁺¹);

    # Newton-Raphson iterations loop
    r₂ = 1.f0; r_prev = Inf; iter = 1;

    while iter < max_iter

        # compute v_{n+1}, a_{n+1}, ... from "Isogeometric analysis: toward integration of CAD and FEA"
        vⁿ⁺¹ = γ/(β*Δt)*dⁿ⁺¹ - γ/(β*Δt)*dⁿ + (1-γ/β)*vⁿ - Δt*(γ/2β-1)*aⁿ;
        aⁿ⁺¹ = 1/(β*Δt^2)*dⁿ⁺¹ - 1/(β*Δt^2)*dⁿ - 1/(β*Δt)*vⁿ - (1/2β-1)*aⁿ;

        # compute d_{n+af}, v_{n+af}, a_{n+am}, ...
        dⁿ⁺ᵅ = αf*dⁿ⁺¹ + (1-αf)*dⁿ;
        vⁿ⁺ᵅ = αf*vⁿ⁺¹ + (1-αf)*vⁿ;
        aⁿ⁺ᵅ = αm*aⁿ⁺¹ + (1-αm)*aⁿ;

        # update the jacobian, the residual and the external force
        integrate!(op, dⁿ⁺ᵅ, force)
        # integrate_inextensible!(op, dⁿ⁺ᵅ, force)

        # compute the jacobian and the residuals
        op.jacob .= αm/(β*Δt^2)*op.mass .+ αf*op.jacob
        op.resid .= op.stiff*dⁿ⁺ᵅ + op.mass*aⁿ⁺ᵅ - op.ext

        # apply BC (the iterate is what puts the Neumann reaction into the residual)
        applyBC!(op, dⁿ⁺ᵅ)

        # check convergence
        r₂ = norm(op)
        # converged, or the iteration has stopped making progress. The second test is not
        # a nicety: `op.resid` is formed from `op.stiff`, which `integrate!` zeroes and
        # rebuilds each pass, so it never carries the Neumann Lagrange coupling that
        # `applyBC!` puts into `op.jacob`. The residual therefore settles on a floor the
        # Newton direction cannot reach, and every further step adds an increment that is
        # invisible to the residual -- it only inflates the multipliers. Without this the
        # loop runs all `max_iter` steps where two do the work.
        (r₂ < tol || r₂ > 0.99r_prev) && break
        r_prev = r₂

        # newton solve for the displacement increment
        dⁿ⁺¹ -= op.jacob\op.resid; iter += 1
    end

    # save variables
    op.u[1] .= dⁿ⁺¹
    op.u[2] .= vⁿ⁺¹
    op.u[3] .= aⁿ⁺¹
end

function solve_step!(op::DynamicFEOperator{Newmark}, force, Δt; tol=1e-4, max_iter=1e3)

    # useful
    β, γ = op.β, op.γ

    # predictor (initial guess) for the Newton-Raphson scheme
    (dⁿ⁺¹, vⁿ⁺¹, aⁿ⁺¹) = op.u

    # predictor step
    dⁿ⁺¹ += Δt*vⁿ⁺¹ + Δt^2/2*(1-2*β)*aⁿ⁺¹;
    vⁿ⁺¹ += Δt*(1-γ)*aⁿ⁺¹;
    aⁿ⁺¹ .= 0.0;

    # Newton-Raphson iterations loop
    r₂ = 1.0; r_prev = Inf; iter = 1;
    while iter < max_iter

        # update the jacobian, the residual and the external force
        integrate!(op, dⁿ⁺¹, force)

        # compute the jacobian and the residuals
        op.jacob .= 1.0/(β*Δt^2)*op.mass .+ op.jacob
        op.resid .= op.stiff*dⁿ⁺¹ + op.mass*aⁿ⁺¹ - op.ext

        # apply BC (the iterate is what puts the Neumann reaction into the residual)
        applyBC!(op, dⁿ⁺¹)

        # check convergence
        r₂ = norm(op)
        # converged, or the iteration has stopped making progress. The second test is not
        # a nicety: `op.resid` is formed from `op.stiff`, which `integrate!` zeroes and
        # rebuilds each pass, so it never carries the Neumann Lagrange coupling that
        # `applyBC!` puts into `op.jacob`. The residual therefore settles on a floor the
        # Newton direction cannot reach, and every further step adds an increment that is
        # invisible to the residual -- it only inflates the multipliers. Without this the
        # loop runs all `max_iter` steps where two do the work.
        (r₂ < tol || r₂ > 0.99r_prev) && break
        r_prev = r₂

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