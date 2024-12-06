using Splines
using LinearAlgebra
using SparseArrays
using Plots

# Material properties and mesh
numElem=4
degP=2
ptLeft = 0.0
ptRight = 1.0
L = 1.0
EI = 0.01
EA = 1_000.0
density(ξ) = 10

# natural frequencies
ωₙ = [1.875, 4.694, 7.855]
fhz = ωₙ.^2.0.*√(EI/(density(0.0)*L^4))/(2π)

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = []

# make a problem
operator = DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC;
                             ρ=density, ρ∞=0.5, I=Newmark)

"""make it inextensible"""
function inextensible!(stiff, resid, x0, op)
    nB,degP = op.mesh.numBasis,op.mesh.degP[1]
    # integrate on each element
    for iElem = 1:op.mesh.numElem
        # compute the (B-)spline basis functions and derivatives with Bezier extraction
        Jac_ref_par,N,dN,ddN = Splines.BSplineBasis(op.mesh, iElem)
            
        # where are we in the stiffness matrix
        # I = element(iElem,degP)
        In = nodes(iElem,degP)

        # integrate on element
        for iGauss = 1:length(op.gauss_rule.nodes)
           #compute the rational basis, this all allpcates at leats one
            RR = N[iGauss,:].* op.mesh.weights[In]
            dR = dN[iGauss,:].* op.mesh.weights[In]
            # ddR = ddN[iGauss,:].* op.mesh.weights[In]
            w_sum = sum(RR)
            dw_xi = sum(dR)
            dR = dR/w_sum - RR*dw_xi/w_sum^2
            # ddR = ddR/w_sum - 2*dR*dw_xi/w_sum^2 - RR*sum(ddR)/w_sum^2 + 2*RR*dw_xi^2/w_sum^3
            #compute the derivatives w.r.t to the physical space
            dxdxi = dR' * op.mesh.controlPoints[1, In]
            Jac_par_phys = det(dxdxi)

            # add the constraint to the stiffness mastrix
            dw0dx = dR' * x0[In.+nB]
            @. stiff[end,In] += Jac_ref_par * Jac_par_phys * dR * op.gauss_rule.weights[iGauss]
            @. stiff[end,(In).+nB] -= 0.5Jac_ref_par * Jac_par_phys * dR * dR * op.gauss_rule.weights[iGauss]
            # inextensible
            resid[end] = 0.0
        end
    end
    # enforce symmetry
    stiff[:,end] = stiff[end,:];
end


# uniform external loading at integration points
force = zeros(2,length(uv_integration(operator))); force[:,:] .= 1.0/5
integrate!(operator,operator.x,force)
applyBC!(operator)

N = size(operator.stiff)
K_inextensible = zeros(N[1]+1,N[2]+1)
K_inextensible[1:N[1],1:N[2]] .= operator.stiff
resid = zeros(N[1]+1)
resid[1:N[1]] .= -operator.ext
x0 = zero(resid)
inextensible!(K_inextensible, resid, x0, operator)


sol = -K_inextensible\resid
# result = x0[1:end-1]
# nlsolve!(operator, result, force; tol=1e-6)
# sol = lsolve!(operator, force)
# sol = result
# get the solution
u0 = sol[1:mesh.numBasis]
w0 = sol[mesh.numBasis+1:2mesh.numBasis]
x = LinRange(ptLeft, ptRight, numElem+1)
u = x .+ getSol(mesh, u0, 1)
w = getSol(mesh, w0, 1)

du = Splines.getDerivSol(mesh, u0)
dw = Splines.getDerivSol(mesh, w0)

# println("beam length ", round.(du + 0.5dw.^2, digits=12))
println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=12))
plot(u,w)

# # time steps
# Δt = 0.1
# T = 1.0/fhz[1]
# tstep = 0.5
# time = [0.]

# # get the results
# xs = LinRange(ptLeft, ptRight, numElem+1)

# # external force
# Nuv = length(uv_integration(operator))
# f_ext = zeros(2,Nuv)

# # time loops
# @time @gif for tᵢ in range(0.,T;step=tstep)
#     t = sum(time)
#     while t < tᵢ
#         f_ext .= 0.0
#         # external loading
#         if t<2.0
#             f_ext[2,Nuv÷2] = -1
#             # f_ext[2, 1] = -1
#             # f_ext[2,Nuv] = 1
#         end
#         # integrate one in time
#         solve_step!(operator, f_ext, Δt)
#         t += Δt; push!(time, Δt)
#     end

#     # get the results
#     u0 = operator.u[1][1:operator.mesh.numBasis]
#     w0 = operator.u[1][operator.mesh.numBasis+1:2operator.mesh.numBasis]
#     u = xs .+ getSol(operator.mesh, u0, 1)
#     w = getSol(operator.mesh, w0, 1)
#     ti =round(t*fhz[1],digits=3)
#     U = [sum(u0)/length(u0),sum(w0)/length(w0)]
#     Plots.plot(u, w, legend=:none, xlim=(-0.5+U[1],1.5+U[1]), aspect_ratio=:equal, 
#                ylims=(-1+U[1],1+U[1]), 
#                title="t = $ti")
#     println("t/T = ", round(t/T,digits=3))
#     println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=4))
# end
