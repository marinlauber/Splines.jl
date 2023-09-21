using Test
using Splines

function test_fixed_fixed_UDL(numElem=2, degP=3)
    
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    f(x) = 1.0
    t(x) = 0.0
    exact_sol(x) = f(x).*x.^2/(24EI).*(1 .- x).^2    # fixed - fixed

    IGAmesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_right = Boundary1D("Dirichlet", ptRight, ptRight, 0.0)
    Dirichlet_left = Boundary1D("Dirichlet", ptLeft, ptLeft, 0.0)
    Neumann_left = Boundary1D("Neumann", ptLeft, ptLeft, 0.0)
    Neumann_right= Boundary1D("Neumann", ptRight, ptRight, 0.0)

    # make a problem
    p = Problem1D(EI, EA, f, t, IGAmesh, gauss_rule,
                 [Dirichlet_left,Dirichlet_right],
                 [Neumann_left,Neumann_right])

    result = static_lsolve!(p, static_residuals!, static_jacobian!)

    u0 = result[1:IGAmesh.numBasis]
    w0 = result[IGAmesh.numBasis+1:2IGAmesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(IGAmesh, u0, 1)
    w = getSol(IGAmesh, w0, 1)
    we = exact_sol(x)
    println("Error: ", norm(w .- we))
    Plots.plot(u, w, label="Sol")
    Plots.plot!(x, we, label="Exact")
end


function test_pinned_pinned_UDL(numElem=2, degP=3)

    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    f(x) = 1.0
    t(x) = 0.0
    exact_sol(x) = f(x)/(24EI).*(x .- 2x.^3 .+ x.^4) # pinned - pinned

    IGAmesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_right = Boundary1D("Dirichlet", ptRight, ptRight, 0.0)
    Dirichlet_left = Boundary1D("Dirichlet", ptLeft, ptLeft, 0.0)

    # make a problem
    p = Problem1D(EI, EA, f, t, IGAmesh, gauss_rule,
                [Dirichlet_left,Dirichlet_right], [])

    # solve the problem
    result = static_lsolve!(p, static_residuals!, static_jacobian!)

    # get the solution
    u0 = result[1:IGAmesh.numBasis]
    w0 = result[IGAmesh.numBasis+1:2IGAmesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(IGAmesh, u0, 1)
    w = getSol(IGAmesh, w0, 1)
    we = exact_sol(x)
    println("Error: ", norm(w .- we))
    Plots.plot(u, w, label="Sol")
    Plots.plot!(x, we, label="Exact")
end


function test_fixed_free_UDL(numElem=2, degP=3)

    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    f(x) = 1.0
    t(x) = 0.0
    exact_sol(x) = f(x).*x.^2/(24EI).*(6 .- 4x .+ x.^2) # fixed - free

    IGAmesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_left = Boundary1D("Dirichlet", ptLeft, ptLeft, 0.0)
    Neumann_left = Boundary1D("Neumann", ptLeft, ptLeft, 0.0)

    # make a problem
    p = Problem1D(EI, EA, f, t, IGAmesh, gauss_rule,
                    [Dirichlet_left], [Neumann_left])

    # solve the problem
    result = static_lsolve!(p, static_residuals!, static_jacobian!)

    # get the results
    u0 = result[1:IGAmesh.numBasis]
    w0 = result[IGAmesh.numBasis+1:2IGAmesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(IGAmesh, u0, 1)
    w = getSol(IGAmesh, w0, 1)
    we = exact_sol(x)
    println("Error: ", norm(w .- we))
    Plots.plot(u, w, label="Sol")
    Plots.plot!(x, we, label="Exact")
end


function test_fixed_free(numElem=2, degP=3)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1
    EA = 12 #/0.1^2
    F = 3EI/2
    println("EI: ", EI, " EA: ", EA, " F: ", F)
    f(x) = 0.0
    t(x) = 0.0

    IGAmesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_left = Boundary1D("Dirichlet", ptLeft, ptLeft, 0.0)
    Neumann_left = Boundary1D("Neumann", ptLeft, ptLeft, 0.0)

    # make a problem
    p = Problem1D(EI, EA, f, t, IGAmesh, gauss_rule,
                    [Dirichlet_left], [Neumann_left])

    # unpack pre-allocated storage and the convergence flag
    @unpack x, resid, jacob, EI, EA, f, t, mesh, gauss_rule, Dirichlet_BC, Neumann_BC = p

    # warp the residual and the jacobian
    function f!(resid, x) 
        static_residuals!(resid, x, f, t, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
        # need a custom solver as the resdiuals are different
        resid[2*IGAmesh.numBasis] -= F
        return resid
    end
    j!(jacob, x) = static_jacobian!(jacob, x, f, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)

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
    resid .= df.F
    jacob .= df.DF

    # get the results
    u0 = result[1:IGAmesh.numBasis]
    w0 = result[IGAmesh.numBasis+1:2IGAmesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(IGAmesh, u0, 1)
    w = getSol(IGAmesh, w0, 1)
    println("u_x = ",2getSol(IGAmesh, u0, 1)[end])
    println("u_y = ",2w[end])
    Plots.plot(u, w, aspect_ratio=:equal, label="Sol")
end

# linear tests
test_fixed_fixed_UDL(3, 4)
test_pinned_pinned_UDL(3, 4)
test_fixed_free_UDL(3, 4)

# non-linear tests, needs validation
test_fixed_free(8, 4)
