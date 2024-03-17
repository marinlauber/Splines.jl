using Splines
using Plots

numElem=2;degP=3

println("Testing on fixed-fixed beam with UDL:")
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
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]

Neumann_BC = []

# make a problem
operator = DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

# uniform external loading at integration points
force = zeros(2,length(uv_integration(operator))); force[2,:] .= 1

# update the jacobian, the residual and the external force
# linearized residuals
integrate!(operator, zero(operator.resid), force)

# compute the residuals
operator.resid .= - operator.ext

# apply BC
applyBC!(operator)

# solve the system and return
result = -operator.jacob\operator.resid


# result = relaxation!(operator,force,[1,5,6,10])

u0 = result[1:mesh.numBasis]
w0 = result[mesh.numBasis+1:2mesh.numBasis]
x = LinRange(ptLeft, ptRight, numElem+1)
u = x .+ getSol(mesh, u0, 1)
w = getSol(mesh, w0, 1)
we = exact_sol(x)
println("Error: ", norm(w .- we))
Plots.plot(u, w, label="Sol")
Plots.plot!(x, we, label="Exact")