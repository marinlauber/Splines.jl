using Splines

# Material properties and mesh
numElem=2
degP=3
ptLeft = 0.0
ptRight = 1.0
EI = 1.0
EA = 1.0
f(x) = [0.0, 0.0]
P = 1.0
# exact_sol(x) = P.*x.^2/(6EI).*(3 .- x) # fixed - free (Ponts Load)
exact_sol(x) = x.^2/(24EI).*(6 .- 4x .+ x.^2) # fixed - free UDL

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions: u(0)=w(0)=0.0, dw(0)/dx=0.0
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = [
    Boundary1D("Neumann", ptLeft, 0.0; comp=1),
    Boundary1D("Neumann", ptLeft, 0.0; comp=2)
]

# make a problem
p = EulerBeam(EI, EA, f, mesh, gauss_rule, Dirichlet_BC, Neumann_BC)

# unpack pre-allocated storage and the convergence flag
@unpack x, resid, jacob = p
stiff = zero(jacob)
mass = zero(jacob)

# mass for UDL
Splines.global_mass!(mass, p.mesh, x->1.0, p.gauss_rule)
gravity = zero(resid); gravity[mesh.numBasis+1:2mesh.numBasis] .= -1.0

# ad hoc BC
mass_bc = copy(mass)
mass_bc[[1,mesh.numBasis+1],:] .= 0.0
mass_bc[:,[1,mesh.numBasis+1]] .= 0.0

# update stiffness and jacobian
Splines.update_global!(stiff, jacob, x, p.mesh, p.gauss_rule, p)
Splines.update_external!(resid, p.mesh, p.f, p.gauss_rule)
resid .+= stiff*x + mass*gravity
# Splines.applyBCNewton!(jacob, resid, p.mesh, p.Dirichlet_BC, p.Neumann_BC, p.gauss_rule)
Splines.applyBCGlobal!(jacob, stiff, resid, p.mesh, p.Dirichlet_BC, p.Neumann_BC, p.gauss_rule)

# solve the problem
x .-= jacob\resid

# recompute residuals
resid .= stiff*x + mass*gravity
# Splines.applyBCNewton!(jacob, resid, p.mesh, p.Dirichlet_BC, p.Neumann_BC, p.gauss_rule)
Splines.applyBCGlobal!(jacob, stiff, resid, p.mesh, p.Dirichlet_BC, p.Neumann_BC, p.gauss_rule)
r = norm(resid)
println("computed after solve $r")


# get the results
u0 = x[1:mesh.numBasis]
w0 = x[mesh.numBasis+1:2mesh.numBasis]
xs = LinRange(ptLeft, ptRight, numElem+1)
u = xs .+ getSol(mesh, u0, 1)
w = getSol(mesh, w0, 1)
we = exact_sol(xs)
println("Error: ", norm(w .- we))
Plots.plot(u, w, label="Sol")
Plots.plot!(xs, we, label="Exact")