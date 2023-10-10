using NLsolve
using ImplicitAD
using LineSearches


function static_nlsolve!(p, residual!, jacobian!)

    # unpack pre-allocated storage and the convergence flag
    # @unpack x, resid, jacob, EI, EA, f, t, mesh, gauss_rule, Dirichlet_BC, Neumann_BC = p
    @unpack x, resid, jacob = p

    # warp the residual and the jacobian
    # f!(resid, x) = residual!(resid, x, f, t, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
    # j!(jacob, x) = jacobian!(jacob, x, f, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
    f!(resid, x) = residual!(resid, x, p)
    j!(jacob, x) = jacobian!(jacob, x, p)

    # prepare for solve
    df = NLsolve.OnceDifferentiable(f!, j!, x, resid, jacob)

    # solve the system
    # result = NLsolve.nlsolve(df, x, method=:newton)

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
    resid .= df.F
    jacob .= df.DF

    return result.zero
end

static_lsolve!(p) = static_lsolve!(p, static_residuals!, static_jacobian!)
function static_lsolve!(p, residual!, jacobian!)

    # unpack pre-allocated storage and the convergence flag
    @unpack x, resid, jacob = p

    # initial consition
    x0 = zeros(length(x))

    # update residual and the jacobian
    residual!(resid, x0, p)
    jacobian!(jacob, x0, p)
    
    # update the state "Newton's method"
    x .= x0 .- ImplicitAD.implicit_linear(jacob, resid)

    return x
end

# function static_residuals!(resid, x0, f, t, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
#     # try subtitute zero initial deflection and contraction
#     u0 = x0[1:mesh.numBasis]
#     w0 = x0[mesh.numBasis:2*mesh.numBasis]

#     # compute matrix
#     K11,K12,K22,J22 = assemble_stiff(mesh, u0, w0, EI, EA, gauss_rule)
#     force = assemble_rhs(mesh, f, gauss_rule)
#     tension = assemble_rhs(mesh, t, gauss_rule)

#     # apply BC
#     K22, rhs22 = applyBCNeumann(K22, force, Neumann_BC, mesh, gauss_rule);
#     K22, rhs22 = applyBCDirichlet(K22, rhs22, Dirichlet_BC, mesh)
    
#     # Dirichlet only on the right-hand-side
#     K11, rhs11 = applyBCDirichlet(K11, tension, Dirichlet_BC, mesh)
#     K12 = applyBCDirichlet(K12, Dirichlet_BC, mesh)

#     # assemble the whole system
#     lhs, rhs = assemble(K11, K12, K22, rhs11, rhs22)

#     # compute residuals
#     resid .= lhs*x0 - rhs
#     return nothing
# end


# function static_jacobian!(jacob, x0, f, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
#     # try subtitute zero initial deflection and contraction
#     u0 = x0[1:mesh.numBasis]
#     w0 = x0[mesh.numBasis:2*mesh.numBasis]

#     # compute matrix
#     K11,K12,K22,J22 = assemble_stiff(mesh, u0, w0, EI, EA, gauss_rule)
#     force = assemble_rhs(mesh, f, gauss_rule)

#     # apply BC
#     K22, rhs22 = applyBCNeumann(K22, force, Neumann_BC, mesh, gauss_rule);
#     K22, rhs22 = applyBCDirichlet(K22, rhs22, Dirichlet_BC, mesh)
#     J22 = applyBCDirichlet(J22, Dirichlet_BC, mesh)

#     # Dirichlet only on the right
#     K11, rhs11 = applyBCDirichlet(K11, force, Dirichlet_BC, mesh)
#     K12 = applyBCDirichlet(K12, Dirichlet_BC, mesh)

#     # assemble the whole system
#     jacob .= Jacobian(K11, K12, K22, J22)
#     return nothing
# end


# function dynamic_residuals!(resid, x0, M, a0, f, t, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
#     # try subtitute zero initial deflection and contraction
#     u0 = w0 = zero(x0[1:mesh.numBasis])

#     # compute matrix
#     K11,K12,K22,J22 = assemble_stiff(mesh, u0, w0, EI, EA, gauss_rule)
#     force = assemble_rhs(mesh, f, gauss_rule)

#     # apply BC
#     K22, rhs22 = applyBCNeumann(K22, force, Neumann_BC, mesh, gauss_rule);
#     K22, rhs22 = applyBCDirichlet(K22, rhs22, Dirichlet_BC, mesh)
#     J22 = applyBCDirichlet(J22, Dirichlet_BC, mesh)

#     # Dirichlet only on the right
#     K11, rhs11 = applyBCDirichlet(K11, force, Dirichlet_BC, mesh)
#     K12 = applyBCDirichlet(K12, Dirichlet_BC, mesh)

#     # assemble the whole system
#     K,F = assemble(K11, K12, K22, rhs11, rhs22)
#     resid .= K*x0 + M*a0 + F
#     return nothing
# end

# function dynamic_mass!(mass, mesh, m₀, Dirichlet_BC, Neumann_BC, gauss_rule)
#     # compute inertia
#     Mi = assemble_mass(mesh, m₀, gauss_rule)
#     r22 = zeros(mesh.numBasis)
#     M22,r22 = applyBCNeumann(Mi, r22, Neumann_BC, mesh, gauss_rule)
#     M22,r22 = applyBCDirichlet(M22, r22, Dirichlet_BC, mesh)
#     M11,r11 = applyBCDirichlet(Mi, zeros(mesh.numBasis), Dirichlet_BC, mesh)
#     M,a0 = assemble(M11, 0*M11, M22, r11, r22);
#     mass .= M
#     return nothing
# end

# function dynamic_nljacobian!()
#     continue
# end

# function dynamic_ljacobian!()
#     continue
# end

# function dynamic_jacobian!(jacob, x0, f, M, α₁, α₂, β, Δt, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
#     # try subtitute zero initial deflection and contraction
#     u0 = x0[1:mesh.numBasis]
#     w0 = x0[mesh.numBasis:2*mesh.numBasis]

#     # compute matrix
#     K11,K12,K22,J22 = assemble_stiff(mesh, u0, w0, EI, EA, gauss_rule)
#     force = assemble_rhs(mesh, f, gauss_rule)

#     # apply BC
#     K22, rhs22 = applyBCNeumann(K22, force, Neumann_BC, mesh, gauss_rule);
#     K22, rhs22 = applyBCDirichlet(K22, rhs22, Dirichlet_BC, mesh)
#     J22 = applyBCDirichlet(J22, Dirichlet_BC, mesh)

#     # Dirichlet only on the right
#     K11, rhs11 = applyBCDirichlet(K11, force, Dirichlet_BC, mesh)
#     K12 = applyBCDirichlet(K12, Dirichlet_BC, mesh)

#     # assemble the whole system
#     K = Jacobian(K11, K12, K22, J22)

#     # for the jacobian
#     jacob .= α₁/(β*Δt^2)*M +  α₂*K
#     return nothing
# end
