using Splines
using LinearAlgebra
using SparseArrays
using Plots

# construct strain operator matrix
function strain_matrix(op)
    Ε = zero(op.stiff)
    nB,degP = op.mesh.numBasis,op.mesh.degP[1]
    # integrate on each element
    for iElem = 1:op.mesh.numElem
        
        # compute the (B-)spline basis functions and derivatives with Bezier extraction
        Jac_ref_par,N,dN,ddN = Splines.BSplineBasis(op.mesh, iElem)
        
        # where are we in the stiffness matrix
        I = element(iElem,degP)
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

            # strain matrix
            Ε[I          ] += Jac_ref_par * Jac_par_phys * (dR*dR') * op.gauss_rule.weights[iGauss]
            Ε[I.+δ(nB,nB)] += Jac_ref_par * Jac_par_phys * (dR*dR') * op.gauss_rule.weights[iGauss]
        end
    end
    return Ε
end

function ∂(mesh::Splines.Mesh, sol0)
    B, dB, _ = Splines.bernsteinBasis([-1.,1.], mesh.degP[1])
    sol = zeros(mesh.numElem+1)
    for iElem in 1:mesh.numElem
        uMin = mesh.elemVertex[iElem, 1]
        uMax = mesh.elemVertex[iElem, 2]
        Jac_ref_par = (uMax-uMin)/2
        curNodes = mesh.elemNode[iElem]
        N_mat = B*mesh.C[iElem]
        dN_mat = dB*mesh.C[iElem]/Jac_ref_par
        solVal = zeros(2)
        for iPlotPt in 1:2
            RR = N_mat[iPlotPt,:].* mesh.weights[curNodes]
            dR = dN_mat[iPlotPt,:].* mesh.weights[curNodes]
            w_sum = sum(RR)
            dw_xi = sum(dR)
            dR = dR/w_sum - RR*dw_xi/w_sum^2
            solVal[iPlotPt] += dR' * sol0[curNodes]
        end
        sol[iElem:iElem+1] .= solVal
    end
    return sol
end

function strains(op::AbstractFEOperator)
    u0 = operator.u[1][1:operator.mesh.numBasis]
    w0 = operator.u[1][operator.mesh.numBasis+1:2operator.mesh.numBasis]
    ∂(op.mesh, u0) .+ 0.5 .* ∂(op.mesh, w0).^2
end

# Material properties and mesh
numElem=8
degP=3
ptLeft = 0.0
ptRight = 1.0
L = 1.0
EI = 0.35e6
EA = 1_000_00.0
density(ξ) = 1

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    # Boundary1D("Dirichlet", ptRight, 0.0; comp=1)
]
Neumann_BC = []

# make a problem
g = 9.81
gravity(i,ξ::T) where T = i==2 ? convert(T,-g) : zero(T)
operator = DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC,
                             g=gravity, ρ=density; ρ∞=0.0, I=Newmark)

# time steps
Δt = 0.001
T = 1
tstep = 0.02
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
        # integrate one in time
        solve_step!(operator, f_ext, Δt; max_iter=1e2)
        t += Δt; push!(time, Δt)
        push!(trace,operator.u[1][2operator.mesh.numBasis])
    end
    print("sum force x ", sum(operator.ext[1:operator.mesh.numBasis]))
    println(" sum force y ", sum(operator.ext[operator.mesh.numBasis+1:2operator.mesh.numBasis]))
    integrate!(operator,operator.u[1],f_ext)
    applyBC!(operator)
    # compute internal loading
    internal = operator.stiff*operator.u[1]
    print("int force x ", sum(internal[1:operator.mesh.numBasis]))
    println(" int force y ", sum(internal[1+operator.mesh.numBasis:2operator.mesh.numBasis]))
    # external force 
    external = operator.mass*operator.u[3]
    print("ext force x ", sum(external[1:operator.mesh.numBasis]))
    println(" ext force y ", sum(external[1+operator.mesh.numBasis:2operator.mesh.numBasis]))
    

    # get the results
    u0 = operator.u[1][1:operator.mesh.numBasis]
    w0 = operator.u[1][operator.mesh.numBasis+1:2operator.mesh.numBasis]
    u = xs .+ getSol(operator.mesh, u0, 1)
    w = getSol(operator.mesh, w0, 1)
    Plots.plot(u, w, legend=:none, xlim=(-0.5,1.5), aspect_ratio=:equal,
               ylims=(-1.75,0.25),title="t = $(round(t,digits=3))")
    println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=8))
end
