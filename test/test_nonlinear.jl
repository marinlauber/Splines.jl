using Splines
using LinearAlgebra
using SparseArrays
using Plots

# Material properties and mesh
numElem=20
degP=2
ptLeft = 0.0
ptRight = 1.0
L = 1.0
EI = 2
EA = 50
density(ξ) = 0.0
α² = 1.0

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = [
    Boundary1D("Neumann", ptLeft, 0.0; comp=1),
    Boundary1D("Neumann", ptLeft, 0.0; comp=2)
]

# make a problem
operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC; ρ=density)
# solver = GeneralizedAlpha(operator; ρ∞=0.5)

# time steps
Nₜ = 10;
println("Number of time increment: ", Nₜ)
# get the results
xs = LinRange(ptLeft, ptRight, numElem+1)

# unpack pre-allocated storage and the convergence flag
@unpack resid, jacob = operator
x = zero(resid)
f = zeros(2,length(uv_integration(operator)))

# time loops
@time @gif for k = 0:Nₜ
    
    # warp the residual and the jacobian
    function f!(resid, x) 
        residual!(resid, x, f, operator)
        # need a custom solver as the resdiuals are different
        resid[2*mesh.numBasis] += π*k/Nₜ*α²*EI
        return nothing
    end
    j!(jacob, x) = jacobian!(jacob, x, f, operator)

    # prepare for solve
    df = NLsolve.OnceDifferentiable(f!, j!, x, resid, jacob)

    # # solve the system
    solver = NLsolve.nlsolve(df, x,
                            show_trace=false,
                            linsolve=(x, A, b) -> x .= ImplicitAD.implicit_linear(A, b),
                            method=:newton,
                            linesearch=LineSearches.BackTracking(maxstep=1e6),
                            ftol=1e-9,
                            iterations=1000)

    # update the state, residual, jacobian, and convergence flag
    result = solver.zero

    # integrate one in time
    # solve_step!(solver, f_ext, Δt)

    # get the results
    # u0 = solver.u[1][1:solver.op.mesh.numBasis]
    # w0 = solver.u[1][solver.op.mesh.numBasis+1:2solver.op.mesh.numBasis]
    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    u = xs .+ getSol(operator.mesh, u0, 21)
    w = getSol(operator.mesh, w0, 21)
    Plots.plot(u, w, legend=:none, xlim=(-0.5*ptRight,ptRight*1.5), aspect_ratio=:equal, ylims=(0.5-2ptRight,0.5), 
                title="tU/L = $k")
    println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=4))
end
