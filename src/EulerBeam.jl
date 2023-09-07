using SparseArrays
using UnPack

struct Problem1D
    x :: Vector{Float64}
    resid :: Vector{Float64}
    jacob :: Matrix{Float64}
    EI    :: Float64
    EA :: Float64
    f :: Function
    t :: Function
    mesh :: Mesh
    gauss_rule :: GaussQuad
    Dirichlet_BC :: Vector{Boundary1D}
    Neumann_BC :: Vector{Boundary1D}
    function Problem1D(EI, EA, f, t, mesh, gauss_rule, Dirichlet_BC=[], Neumann_BC=[])
        numNeuBC = length(Neumann_BC)
        # generate
        x = zeros(Float64, 2*mesh.numBasis+numNeuBC)
        resid = zeros(Float64, 2*mesh.numBasis+numNeuBC)
        jacob = spzeros(2*mesh.numBasis+numNeuBC, 2*mesh.numBasis+numNeuBC)
        rhs = zeros(Float64, 2*mesh.numBasis+numNeuBC)
        new(x, resid, jacob, EI, EA, f, t, mesh, gauss_rule, Dirichlet_BC, Neumann_BC)
    end
end 

"""
Aseembles the stiffness matrix for 2D Elasticity Kₑ = ∫ B^T*C*B dΩ
"""
function assemble_stiff(mesh::Mesh, EI, EA, gauss_rule)
    w0 = zeros(Float64, mesh.numBasis)
    return assemble_stiff(mesh, w0, w0, EI, EA, gauss_rule)
end

"""
"""
function assemble_stiff(mesh::Mesh, u0, w0, EI, EA, gauss_rule)
    B, dB, ddB = bernsteinBasis(gauss_rule.nodes, mesh.degP[1])
    domainLength = 0
    K22 = spzeros(mesh.numBasis, mesh.numBasis)
    K11 = spzeros(mesh.numBasis, mesh.numBasis)
    K12 = spzeros(mesh.numBasis, mesh.numBasis)
    J22 = spzeros(mesh.numBasis, mesh.numBasis)
    for iElem = 1:mesh.numElem
        uMin = mesh.elemVertex[iElem, 1]
        uMax = mesh.elemVertex[iElem, 2]
        Jac_ref_par = (uMax-uMin)/2

        #compute the (B-)spline basis functions and derivatives with Bezier extraction
        N_mat = B * mesh.C[iElem]'
        dN_mat = dB * mesh.C[iElem]'/Jac_ref_par
        ddN_mat = ddB * mesh.C[iElem]'/Jac_ref_par^2

        #compute the rational spline basis
        curNodes = mesh.elemNode[iElem]
        numNodes = length(curNodes)
        cpts = mesh.controlPoints[1, curNodes]
        wgts = mesh.weights[curNodes]
        k11 = zeros(numNodes, numNodes)
        k12 = zeros(numNodes, numNodes)
        k22 = zeros(numNodes, numNodes)
        j22 = zeros(numNodes, numNodes)
        for iGauss = 1:length(gauss_rule.nodes)
            #compute the rational basis
            RR = N_mat[iGauss,:].* wgts
            dR = dN_mat[iGauss,:].* wgts
            ddR = ddN_mat[iGauss,:].* wgts
            w_sum = sum(RR)
            dw_xi = sum(dR)
            dR = dR/w_sum - RR*dw_xi/w_sum^2
            ddR = ddR/w_sum - 2*dR*dw_xi/w_sum^2 - RR*sum(ddR)/w_sum^2 + 2*RR*dw_xi^2/w_sum^3

            #compute the derivatives w.r.t to the physical space
            dxdxi = dR' * cpts
            Jac_par_phys = det(dxdxi)

            # compute linearised terms using the current solution
            du0dx = dR' * u0[curNodes]
            dw0dx = dR' * w0[curNodes]

            # compute the different terms
            k11 += Jac_ref_par * Jac_par_phys * EA * (dR*dR') * gauss_rule.weights[iGauss]
            k12 += 0.5 * Jac_ref_par * Jac_par_phys * EA * (dw0dx) * (dR*dR') * gauss_rule.weights[iGauss]
            k22 += Jac_ref_par * Jac_par_phys^2 * EI * (ddR*ddR') * gauss_rule.weights[iGauss]
            k22 += 0.5 * Jac_ref_par * Jac_par_phys * EA * (du0dx + dw0dx^2) * (dR*dR') * gauss_rule.weights[iGauss]
            
            # only different entry of Jacobian
            j22 += Jac_ref_par * Jac_par_phys * EA* (dw0dx^2) * (dR*dR') * gauss_rule.weights[iGauss]
            # check if the domain is correct
            domainLength += Jac_ref_par * Jac_par_phys * gauss_rule.weights[iGauss]
        end
        K11[curNodes, curNodes] += k11
        K12[curNodes, curNodes] += k12
        K22[curNodes, curNodes] += k22
        J22[curNodes, curNodes] += j22
    end
    # @show domainLength
    # symmetrized case K21 = K12
    return K11,K12,K22,J22
end

"""
Aseembles the mass matrix M_ij = ∫ a1(x)ϕ_i(x)ϕ_j(x) dΩ
"""
function assemble_mass(mesh::Mesh, m, gauss_rule)
    B, dB = bernsteinBasis(gauss_rule.nodes, mesh.degP[1])
    domainLength = 0
    mass = spzeros(Float64, mesh.numBasis, mesh.numBasis)
    for iElem = 1:mesh.numElem
        uMin = mesh.elemVertex[iElem, 1]
        uMax = mesh.elemVertex[iElem, 2]
        Jac_ref_par = (uMax-uMin)/2

        #compute the (B-)spline basis functions and derivatives with Bezier extraction
        N_mat = B * mesh.C[iElem]'
        dN_mat = dB * mesh.C[iElem]'/Jac_ref_par

        #compute the rational spline basis
        curNodes = mesh.elemNode[iElem]
        numNodes = length(curNodes)
        cpts = mesh.controlPoints[1, curNodes]
        wgts = mesh.weights[curNodes]
        localMass = zeros(numNodes, numNodes)
        for iGauss = 1:length(gauss_rule.nodes)
            #compute the rational basis
            RR = N_mat[iGauss,:].* wgts
            dR = dN_mat[iGauss,:].* wgts
            w_sum = sum(RR)
            dw_xi = sum(dR)
            dR = dR/w_sum - RR*dw_xi/w_sum^2

            #compute the Jacobian of the transformation from parameter to physical space
            dxdxi = dR' * cpts
            Jac_par_phys = det(dxdxi)

            RR /= w_sum
            localMass += m * (RR*RR') * Jac_par_phys * Jac_ref_par * gauss_rule.weights[iGauss]
            domainLength += Jac_par_phys * Jac_ref_par * gauss_rule.weights[iGauss]

        end
        #@show localMass
        #readline(stdin)
        mass[curNodes, curNodes] += localMass
    end
    # @show domainLength
    return mass
end

"""
Assembles the RHS vector corresponding to the body force
RHS[i]=∫_Ω ϕ_i(x)*f(x) dΩ
"""
function assemble_rhs(mesh::Mesh, f::Function, gauss_rule)
    B, dB, ddB = bernsteinBasis(gauss_rule.nodes, mesh.degP[1])
    domainLength = 0
    rhs = zeros(Float64, mesh.numBasis)
    for iElem = 1:mesh.numElem
        uMin = mesh.elemVertex[iElem, 1]
        uMax = mesh.elemVertex[iElem, 2]
        Jac_ref_par = (uMax-uMin)/2
        #@show Jac_ref_par
        #compute the (B-)spline basis functions and derivatives with Bezier extraction
        N_mat = B * mesh.C[iElem]'
        dN_mat = dB * mesh.C[iElem]'/Jac_ref_par

        #compute the rational spline basis
        curNodes = mesh.elemNode[iElem]
        numNodes = length(curNodes)
        cpts = mesh.controlPoints[1, curNodes]
        wgts = mesh.weights[curNodes]
        localRhs = zeros(numNodes)
        for iGauss = 1:length(gauss_rule.nodes)
            #compute the rational basis
            RR = N_mat[iGauss,:].* wgts
            dR = dN_mat[iGauss,:].* wgts
            w_sum = sum(RR)
            dw_xi = sum(dR)
            dR = dR/w_sum - RR*dw_xi/w_sum^2

            #compute the derivatives w.r.t to the physical space
            dxdxi = dR' * cpts
            Jac_par_phys = det(dxdxi)
            RR /= w_sum

            phys_pt = RR'*cpts
            localRhs += Jac_par_phys * Jac_ref_par * f(phys_pt) * RR * gauss_rule.weights[iGauss]
            domainLength += Jac_par_phys * Jac_ref_par * gauss_rule.weights[iGauss]
        end
        rhs[curNodes] += localRhs
    end
    # @show domainLength
    return rhs
end


function assemble(K11, K12, K22, r11, r22)
    K = spzeros(size(K11,1)+size(K22,1), size(K11,2)+size(K22,2))
    R = zeros(size(r11,1)+size(r22,1))
    K[1:size(K11,1),1:size(K11,2)] .= K11
    K[end-size(K22,1)+1:end,end-size(K22,2)+1:end] .= K22
    K[1:size(K12,1),size(K11,2)+1:size(K11,2)+size(K12,2)] .= K12
    K[size(K11,1)+1:size(K11,1)+size(K12,1),1:size(K12,2)] .= K12'
    R[1:size(r11,1)] .= r11
    R[end-size(r22,1)+1:end] .= r22
    return K,R
end

function Jacobian(K11,K12,K22,J22)
    J = spzeros(size(K11,1)+size(K22,1), size(K11,2)+size(K22,2))
    J[1:size(K11,1),1:size(K11,2)] .= K11
    J[1:size(K12,1),size(K11,2)+1:size(K11,2)+size(K12,2)] .= K12
    J[size(K11,1)+1:size(K11,1)+size(K12,1),1:size(K12,2)] .= K12'
    J[end-size(K22,1)+1:end,end-size(K22,2)+1:end] .= K22
    dx=size(K22,1)-size(J22,1); dy=size(K22,2)-size(J22,2)
    J[end-size(K22,1)+1:end-dx,end-size(K22,2)+1:end-dy] .+= J22
    return J
end


function getDerivSol(mesh::Mesh, sol0)
    numPtsElem = 4
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB, ddB = bernsteinBasis(evalPts, mesh.degP[1])
    sol = zeros((numPtsElem-1)*mesh.numElem+1)
    is=1; ie=numPtsElem
    for iElem in 1:mesh.numElem
        uMin = mesh.elemVertex[iElem, 1]
        uMax = mesh.elemVertex[iElem, 2]
        Jac_ref_par = (uMax-uMin)/2
        curNodes = mesh.elemNode[iElem]
        N_mat = B*(mesh.C[iElem])'
        dN_mat = dB * mesh.C[iElem]'/Jac_ref_par
        wgts = mesh.weights[curNodes]
        solVal = zeros(numPtsElem)
        for iPlotPt = 1:numPtsElem
            RR = N_mat[iPlotPt,:].* wgts
            dR = dN_mat[iPlotPt,:].* wgts
            w_sum = sum(RR)
            dw_xi = sum(dR)
            dR = dR/w_sum - RR*dw_xi/w_sum^2
            solVal[iPlotPt] += dR' * sol0[curNodes]
        end
        sol[is:ie] .= solVal
        is += numPtsElem-1 ; ie += numPtsElem-1
    end
    return sol
end

function getSol(mesh::Mesh, sol0, numPts=mesh.numElem)
    numPtsElem = max(floor(Int, numPts/mesh.numElem)+1,2)
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB, ddB = bernsteinBasis(evalPts, mesh.degP[1])
    sol = zeros((numPtsElem-1)*mesh.numElem+1)
    is=1; ie=numPtsElem
    for iElem in 1:mesh.numElem
        curNodes = mesh.elemNode[iElem]
        splineVal = B*(mesh.C[iElem])'
        wgts = mesh.weights[curNodes]
        basisVal = zero(splineVal)
        for iPlotPt = 1:numPtsElem
            RR = splineVal[iPlotPt,:].* wgts
            w_sum = sum(RR)
            RR /= w_sum
            basisVal[iPlotPt,:] = RR
        end
        solVal = basisVal*sol0[curNodes]
        sol[is:ie] .= solVal
        is += numPtsElem-1 ; ie += numPtsElem-1
    end
    return sol
end

function applyBCDirichlet(lhs, rhs, bound_cond, mesh)
    for i=eachindex(bound_cond)
        bcdof = Array{Int64,1}(undef, 0)
        bcval = Array{Float64,1}(undef, 0)
        bcdof = vcat(bcdof, findall(mesh.controlPoints[1,:].==bound_cond[i].x_val))
        bcval = vcat(bcval, bound_cond[i].op_val)
        rhs = rhs - lhs[:,bcdof]*bcval
        rhs[bcdof] = bcval
        lhs[bcdof, :] .= 0.
        lhs[:, bcdof] .= 0.
        lhs[bcdof, bcdof] = sparse(I, length(bcdof), length(bcdof))
    end
    return lhs, rhs
end

# strikes trough the LHS where the BC matches the index
function applyBCDirichlet(lhs, bound_cond, mesh)
    for i=eachindex(bound_cond)
        bcdof = Array{Int64,1}(undef, 0)
        bcdof = vcat(bcdof, findall(mesh.controlPoints[1,:].==bound_cond[i].x_val))
        lhs[bcdof, :] .= 0.
        lhs[:, bcdof] .= 0.
    end
    return lhs
end

function applyBCNeumann(stiff, f, bound_cond, mesh, gauss_rule)
    Bmat = spzeros(Float64, length(bound_cond), mesh.numBasis)
    rhs = zeros(Float64, size(stiff,1)+length(bound_cond))
    for i=eachindex(bound_cond)
        cdof_neu = findall(mesh.controlPoints[1,:].==bound_cond[i].x_val)
        for iElem=1:mesh.numElem
            curNodes = mesh.elemNode[iElem]
            if cdof_neu[1] in curNodes
                # nodes = gauss_rule.nodes
                nodes = [2*bound_cond[i].u_val-1] # map into parameter space
                # compute the Bernstein basis functions and derivatives
                B, dB, ddB = bernsteinBasis(nodes, mesh.degP[1])
                uMin = mesh.elemVertex[iElem, 1]
                uMax = mesh.elemVertex[iElem, 2]
                Jac_ref_par = (uMax-uMin)/2

                #compute the (B-)spline basis functions and derivatives with Bezier extraction
                N_mat = B * mesh.C[iElem]'
                dN_mat = dB * mesh.C[iElem]'/Jac_ref_par

                #compute the rational spline basis
                numNodes = length(curNodes)
                cpts = mesh.controlPoints[1, curNodes]
                wgts = mesh.weights[curNodes]
                localLagrange = zeros(numNodes)
                for iGauss = 1:length(nodes)
                    #compute the rational basis
                    RR = N_mat[iGauss,:].* wgts
                    dR = dN_mat[iGauss,:].* wgts
                    w_sum = sum(RR)
                    dw_xi = sum(dR)
                    dR = dR/w_sum - RR*dw_xi/w_sum^2

                    #compute the Jacobian of the transformation from parameter to physical space
                    dxdxi = dR' * cpts
                    Jac_par_phys = det(dxdxi)

                    localLagrange += Jac_ref_par * Jac_par_phys * dR * gauss_rule.weights[iGauss]
                end
                Bmat[i,curNodes] += localLagrange
                rhs[mesh.numBasis+i] = bound_cond[i].op_val
            end
        end
    end
    rhs[1:size(stiff,1)] .= f
    lhs = spzeros(size(stiff,1)+length(bound_cond), size(stiff,2)+length(bound_cond))
    lhs[1:size(stiff,1),1:size(stiff,2)] .= sparse(stiff)
    lhs[end-size(Bmat,1)+1:end,1:size(Bmat,2)] .= Bmat
    lhs[1:size(Bmat,2),end-size(Bmat,1)+1:end] .= Bmat'
    return lhs, rhs
end

function getBasis(mesh::Mesh)
    numPtsElem = 11
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB, ddB = bernsteinBasis(evalPts, mesh.degP[1])
    R = zeros((numPtsElem-1)*mesh.numElem+1, mesh.numBasis)
    dR = zeros((numPtsElem-1)*mesh.numElem+1, mesh.numBasis)
    for iBasis = 1:mesh.numBasis
        for iElem = 1:mesh.numElem
            localIndex = findall(isequal(iBasis), mesh.elemNode[iElem])
            if length(localIndex)>0
                plotVal = B*(mesh.C[iElem][localIndex,:])'
                R[(iElem-1)*(numPtsElem-1)+1:iElem*(numPtsElem-1)+1, iBasis] .= plotVal
                plotVal = dB*(mesh.C[iElem][localIndex,:])'
                dR[(iElem-1)*(numPtsElem-1)+1:iElem*(numPtsElem-1)+1, iBasis] .= plotVal
            end
        end
    end
    return R,dR
end

function getBasis(mesh::Mesh, x::Float64)
    iElem = argmin(abs.(sum(mesh.elemVertex.-x, dims=2))[:])
    points = mesh.elemVertex[iElem,:]
    evalPts = [(x-points[1])/(points[2]-points[1])*2-1]
    B, dB, ddB = bernsteinBasis(evalPts, mesh.degP[1])
    R = zeros(mesh.numBasis)
    dR = zeros(mesh.numBasis)
    for iBasis = 1:mesh.numBasis
        localIndex = findall(isequal(iBasis), mesh.elemNode[iElem])
        if length(localIndex)>0
            plotVal = B*mesh.C[iElem][localIndex,:]'
            R[iBasis] = plotVal[1]
            plotVal = dB*mesh.C[iElem][localIndex,:]'
            dR[iBasis] = plotVal[1]
        end
    end
    return R,dR
end
