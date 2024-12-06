using Splines
using LinearAlgebra
using SparseArrays
using Plots

# Mesh property
numElem=3
degP=3
ptLeft = 0.0
ptRight = 1.0
# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = []

# make a structure
g = 9.81
gravity(i,ξ::T) where T = i==2 ? convert(T,-g) : zero(T)
density(ξ) = 1.0
operator = DynamicFEOperator(mesh, gauss_rule, 1.0, 10_000_000, 
                             Dirichlet_BC, Neumann_BC, ρ=density;
                             ρ∞=0.5, I=GeneralizedAlpha)

# time steps
Δt = 0.01
T = 120.0
tstep = 2.0
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
        f_ext[2,:] .= -g
        # integrate one in time
        solve_step!(operator, f_ext, Δt; max_iter=50)
        t += Δt; push!(time, Δt)
    end
    # @show operator

    # get the results
    u0 = operator.u[1][1:operator.mesh.numBasis]
    w0 = operator.u[1][operator.mesh.numBasis+1:2operator.mesh.numBasis]
    u = xs .+ getSol(operator.mesh, u0, 1)
    w = getSol(operator.mesh, w0, 1)
    U = [sum(u0)/length(u0),sum(w0)/length(w0)]
    Plots.plot(u, w, legend=:none, aspect_ratio=:equal, 
               title="t = $(round(t,digits=3))")
    # # fx = getSol(operator.mesh, operator.ext[1:operator.mesh.numBasis], 1)
    # # fy = getSol(operator.mesh, operator.ext[operator.mesh.numBasis+1:2operator.mesh.numBasis], 1)
    # fx = getSol(operator.mesh, operator.u[3][1:operator.mesh.numBasis], 1)
    # fy = getSol(operator.mesh, operator.u[3][operator.mesh.numBasis+1:2operator.mesh.numBasis], 1)
    xlims!(-1,1)
    ylims!(-1.75,0.25)
    # quiver!(u, w, quiver=(fx*-5e4, fy*-5e4))
    println("t/T = ", round(t/T,digits=3))
    println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=4))
end
