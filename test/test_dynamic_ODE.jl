using Splines
using LinearAlgebra
using SparseArrays
using Plots

# Material properties and mesh
numElem=2
degP=3
ptLeft = 0.0
ptRight = 1.0
L = 1.0
EI = 0.35
EA = 1e5
density(ξ) = 0.0
P = 3EI/2

# natural frequencies
ωₙ = [1.875, 4.694, 7.855]
fhz = ωₙ.^2.0.*√(EI/(density(0.0)*L^4))/(2π)
fhz[1] = 0.25
@show 1.0./fhz

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
solver = GeneralizedAlpha(operator; ρ∞=0.5)

# time steps
Δt = 0.2
T = 10*L/fhz[1]
time = collect(0.0:Δt:T);
Nₜ = length(time);
println("Number of time steps: ", Nₜ)
# get the results
xs = LinRange(ptLeft, ptRight, numElem+1)

# external force
f_ext = zeros(2,length(uv_integration(solver.op)))

# time loops
@time @gif for k = 2:Nₜ
    # external loading
    f_ext .= 0.0
    f_ext[2,:] .= 2P*sin(2π*fhz[1]*time[k]/L)
    
    # integrate one in time
    solve_step!(solver, f_ext, Δt)

    # get the results
    u0 = solver.u[1][1:solver.op.mesh.numBasis]
    w0 = solver.u[1][solver.op.mesh.numBasis+1:2solver.op.mesh.numBasis]
    u = xs .+ getSol(solver.op.mesh, u0, 1)
    w = getSol(solver.op.mesh, w0, 1)
    ti =round(time[k],digits=3)
    Plots.plot(u, w, legend=:none, xlim=(-0.5*ptRight,ptRight*1.5), aspect_ratio=:equal, ylims=(-ptRight,ptRight), 
               title="tU/L = $(ti/L)")
    println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=4))
end
