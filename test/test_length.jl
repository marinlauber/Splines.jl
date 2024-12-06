using Splines
using Plots

numElem=2
degP=3

println("Testing on fixed-fixed beam with UDL:")
println(" numElem: ", numElem)
println(" degP: ", degP)
# Material properties and mesh
ptLeft = 0.0
ptRight = 10.0
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
force = zeros(2,length(uv_integration(operator))); force[2,:] .= 1.0*ptRight
result = lsolve!(operator, force)

u0 = result[1:mesh.numBasis]
w0 = result[mesh.numBasis+1:2mesh.numBasis]
x = LinRange(ptLeft, ptRight, numElem+1)
u = x .+ ptRight*getSol(mesh, u0, 1)
w = ptRight*getSol(mesh, w0, 1)
we = exact_sol(x/ptRight)
println("Error: ", norm(w .- we))
Plots.plot(u, w, label="Sol")
Plots.plot!(u, we*ptRight, label="Exact")
