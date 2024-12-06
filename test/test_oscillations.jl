using Splines
using LinearAlgebra
using SparseArrays
using Plots

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0
L = 1.0
EI = 0.35
EA = 100.0
density(ξ) = 5

# natural frequencies
ωₙ = [1.875, 4.694, 7.855]
fhz = ωₙ.^2.0.*√(EI/(density(0.0)*L^4))/(2π)

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
operator = DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC;
                             ρ=density, ρ∞=0.5, I=GeneralizedAlpha)

# time steps
Δt = 0.05
T = 5.0/fhz[1]
tstep = 0.5
time = [0.]

# get the results
xs = LinRange(ptLeft, ptRight, numElem+1)

# external force
f_ext = zeros(2,length(uv_integration(operator)))
trace = []

# time loops
@time @gif for tᵢ in range(0.,T;step=tstep)
    t = sum(time)
    while t < tᵢ
        f_ext .= 0.0
        # external loading
        if t<2.0
            f_ext[2,2operator.mesh.numBasis] = -4
        end
        f_ext[1,2operator.mesh.numBasis] = 0.5
        # integrate one in time
        solve_step!(operator, f_ext, Δt)
        t += Δt; push!(time, Δt)
        push!(trace,operator.u[1][2operator.mesh.numBasis])
    end

    # get the results
    u0 = operator.u[1][1:operator.mesh.numBasis]
    w0 = operator.u[1][operator.mesh.numBasis+1:2operator.mesh.numBasis]
    u = xs .+ getSol(operator.mesh, u0, 1)
    w = getSol(operator.mesh, w0, 1)
    ti =round(t*fhz[1],digits=3)
    Plots.plot(u, w, legend=:none, xlim=(-0.5,1.5), aspect_ratio=:equal, ylims=(-1,1), 
               title="t = $ti")
    println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=8))
end
# trace_10_000_000 = trace
# # trace_1_000_000 = trace
# # trace_500_000 = trace
# # trace_100_000 = trace
# # trace_100 = trace
# plot(trace_100,label="EI: 0.35, EA: 100");
# plot!(trace_100_000,label="EI: 0.35, EA: 100_000");
# plot!(trace_500_000,label="EI: 0.35, EA: 500_000");
# plot!(trace_1_000_000,label="EI: 0.35, EA: 1_000_000");
# plot!(trace_10_000_000,label="EI: 0.35, EA: 10_000_000")
# xlabel!("Time"); ylabel!("δₜᵢₚ/L"); xlims!(0,600); ylims!(-0.5,0.5)
# savefig("trace_different_EAoverEI.png")