using Splines
using LinearAlgebra
using SparseArrays
using Plots

# Material properties and mesh
numElem=3
degP=3
ptLeft = 0.0
ptRight = 1.0
L = 1.0
EI = 0.001
EA = 1_000_000.0
density(ξ) = 10

# natural frequencies
ωₙ = [1.875, 4.694, 7.855]
fhz = ωₙ.^2.0.*√(EI/(density(0.0)*L^4))/(2π)

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = []
Neumann_BC = []

# make a problem
operator = DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC;
                             ρ=density, ρ∞=0.5, I=Newmark)

# time steps
Δt = 0.1
T = 1/fhz[1]
tstep = 0.5
time = [0.]

# get the results
xs = LinRange(ptLeft, ptRight, numElem+1)

# external force
Nuv = length(uv_integration(operator))
f_ext = zeros(2,Nuv)

# time loops
@time @gif for tᵢ in range(0.,T;step=tstep)
    t = sum(time)
    while t < tᵢ
        f_ext .= 0.0
        # external loading
        if t<1.0/fhz[1]
            # f_ext[2,Nuv÷2] = -1
            f_ext[:, 1] .= -0.1
            # f_ext[2,Nuv] = 1
        end
        # integrate one in time
        solve_step!(operator, f_ext, Δt)
        t += Δt; push!(time, Δt)
    end
    @show operator.u[3]
    @show f_ext

    # get the results
    u0 = operator.u[1][1:operator.mesh.numBasis]
    w0 = operator.u[1][operator.mesh.numBasis+1:2operator.mesh.numBasis]
    u = xs .+ getSol(operator.mesh, u0, 1)
    w = getSol(operator.mesh, w0, 1)
    ti =round(t*fhz[1],digits=3)
    U = [sum(u0)/length(u0),sum(w0)/length(w0)]
    Plots.plot(u, w, legend=:none, xlim=(-0.5+U[1],1.5+U[1]), aspect_ratio=:equal, 
            #    ylims=(-1+U[1],1+U[1]), 
               title="t = $ti")
    println("t/T = ", round(t/T,digits=3))
    println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=4))
end
