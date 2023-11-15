using Test
using Splines

# @testset "bernstein.jl" begin
#     @test bernsteinBasis(1, 4) ≈ [[0.,0.,0.,1],[],[]]
# end


function test_fixed_fixed_UDL(numElem=2, degP=3)

    println("Testing on fixed-fixed beam with UDL:")
    println(" numElem: ", numElem)
    println(" degP: ", degP)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    exact_sol(x) = x.^2/(24EI).*(1 .- x).^2    # fixed - fixed

    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
        Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [
        Boundary1D("Neumann", ptLeft, 0.0; comp=2),
        Boundary1D("Neumann", ptRight, 0.0; comp=2)
    ]

    # make a problem
    operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # uniform external loading at integration points
    force = zeros(2,length(uv_integration(operator))); force[2,:] .= 1.0
    result = lsolve!(operator, force)

    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    we = exact_sol(x)
    println("Error: ", norm(w .- we))
    Plots.plot(u, w, label="Sol")
    Plots.plot!(x, we, label="Exact")
end


function test_pinned_pinned_UDL(numElem=2, degP=3)
    println("Testing on pinned-pinned beam with UDL:")
    println(" numElem: ", numElem)
    println(" degP: ", degP)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    exact_sol(x) = 1.0/(24EI).*(x .- 2x.^3 .+ x.^4) # pinned - pinned

    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
        Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = []

    # make a problem
    operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # uniform external loading at integration points
    force = zeros(2,length(uv_integration(operator))); force[2,:] .= 1.0
    result = lsolve!(operator, force)

    # get the solution
    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    we = exact_sol(x)
    println(" Error: ", norm(w .- we))
    Plots.plot(u, w, label="Sol")
    Plots.plot!(x, we, label="Exact")
end


function test_fixed_free_UDL(numElem=2, degP=3)
    println("Testing on fixed-free beam with UDL:")
    println(" numElem: ", numElem)
    println(" degP: ", degP)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    exact_sol(x) = x.^2/(24EI).*(6 .- 4x .+ x.^2) # fixed - free

    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [Boundary1D("Neumann", ptLeft, 0.0; comp=2)]

    # make a problem
    operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # uniform external loading at integration points
    force = zeros(2,length(uv_integration(operator))); force[2,:] .= 1.0
    result = lsolve!(operator, force)

    # get the results
    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    we = exact_sol(x)
    println(" Error: ", norm(w .- we))
    Plots.plot(u, w, label="Sol")
    Plots.plot!(x, we, label="Exact")
end


function test_fixed_free_PL(numElem=2, degP=3)
    println("Testing on fixed-free beam with Point Load:")
    println(" numElem: ", numElem)
    println(" degP: ", degP)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    P = 1.0
    exact_sol(x) = P.*x.^2/(6EI).*(3 .- x) # fixed - free (Ponts Load)

    # mesh
    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions: u(0)=w(0)=0.0, dw(0)/dx=0.0
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [
        Boundary1D("Neumann", ptLeft, 0.0; comp=2)
    ]

    # make a problem
    operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # linearized residuals
    integrate!(operator, zero(operator.resid), zeros(2,length(uv_integration(operator))))

    # compute the residuals
    operator.resid .= - operator.ext

    # apply BC
    applyBC!(operator)

    # solve static problem with point load
    operator.resid[2*mesh.numBasis] -= P

    # solve the problem
    result = -operator.jacob\operator.resid

    # get the results
    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    we = exact_sol(x)
    println(" Error: ", norm(w .- we))
    Plots.plot(u, w, label="Sol")
    Plots.plot!(x, we, label="Exact")
end


function test_fixed_free(numElem=2, degP=3)
    println("Testing on fixed-free beam non-linear:")
    println(" numElem: ", numElem)
    println(" degP: ", degP)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1
    EA = 12 #/0.1^2
    F = 3EI/2
    println(" EI: ", EI, " EA: ", EA, " F: ", F)
    f(x) = [0.0, 0.0]

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
    operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # unpack pre-allocated storage and the convergence flag
    @unpack resid, jacob = operator
    f = zeros(2,length(uv_integration(operator)))
    x = zero(resid)

    # warp the residual and the jacobian
    function f!(resid, x) 
        residual!(resid, x, f, operator)
        # need a custom solver as the resdiuals are different
        resid[2*mesh.numBasis] -= F
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
    # resid .= df.F
    # jacob .= df.DF

    # get the results
    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    println("u_x = ",getSol(mesh, u0, 1)[end])
    println("u_y = ",w[end])
    println("Error in length :", sum(.√(diff(u).^2 .+ diff(w).^2)))
    Plots.plot(u, w, aspect_ratio=:equal, label="Sol")
end

# linear tests
test_fixed_fixed_UDL(3, 4)
test_pinned_pinned_UDL(3, 4)
test_fixed_free_UDL(3, 4)
test_fixed_free_PL(3, 3)

# # # non-linear tests, needs validation
test_fixed_free(8, 4)
