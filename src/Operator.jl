using SparseArrays
using UnPack

# Abstract FEA operaotr
abstract type AbstractFEOperator end

struct FEOperator <: AbstractFEOperator
    x :: Vector{Float64}
    resid :: Vector{Float64}
    stiff :: Matrix{Float64}
    jacob :: Matrix{Float64}
    ext   :: Vector{Float64}
    mesh :: Mesh
    gauss_rule :: GaussQuad
    BC :: Vector{Boundary1D}
    EI :: Float64
    EA :: Float64
    function FEOperator(mesh::Mesh, gauss_rule::GaussQuad, BC::Vector{Boundary1D})
        numNodes = mesh.numNodes
        numBasis = mesh.numBasis
        new(zeros(2*numBasis), zeros(2*numBasis), zeros(2*numBasis, 2*numBasis), 
            zeros(2*numBasis, 2*numBasis), zeros(2*numBasis), mesh, gauss_rule, BC)
    end
end

function integrate!(op::FEOperator, x0::Vector{Float64}, fx)
    off = op.mesh.numBasis

    op.stiff[1:2off,1:2off] .= 0.;
    op.jacob[1:2off,1:2off] .= 0.;

    B, dB, ddB = bernsteinBasis(op.gauss_rule.nodes, op.mesh.degP[1])
    domainLength = 0
    for iElem = 1:mesh.numElem
        uMin = op.mesh.elemVertex[iElem, 1]
        uMax = op.mesh.elemVertex[iElem, 2]
        Jac_ref_par = (uMax-uMin)/2

        #compute the (B-)spline basis functions and derivatives with Bezier extraction
        N_mat = B * op.mesh.C[iElem]'
        dN_mat = dB * op.mesh.C[iElem]'/Jac_ref_par
        ddN_mat = ddB * op.mesh.C[iElem]'/Jac_ref_par^2

        #compute the rational spline basis
        curNodes = op.mesh.elemNode[iElem]
        cpts = op.mesh.controlPoints[1, curNodes]
        wgts = op.mesh.weights[curNodes]

        # local external force
        localx = zeros(numNodes)
        localy = zeros(numNodes)

        # integrate on element
        for iGauss = 1:length(op.gauss_rule.nodes)
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
            du0dx = dR' * x0[curNodes]
            dw0dx = dR' * x0[curNodes.+off]

            # external force at physical point
            fi = fx[:,(iElem-1)*op.mesh.numElem+iGauss]
            localx += Jac_par_phys * Jac_ref_par * fi[1] * RR * op.gauss_rule.weights[iGauss]
            localy += Jac_par_phys * Jac_ref_par * fi[2] * RR * op.gauss_rule.weights[iGauss]

            # compute the different terms
            op.stiff[curNodes, curNodes] += Jac_ref_par * Jac_par_phys * op.EA * (dR*dR') * op.gauss_rule.weights[iGauss]
            op.stiff[curNodes, curNodes.+off] += 0.5 * Jac_ref_par * Jac_par_phys * op.EA * (dw0dx) * (dR*dR') * op.gauss_rule.weights[iGauss]
            op.stiff[curNodes.+off, curNodes.+off] += Jac_ref_par * Jac_par_phys^2 * op.EI * (ddR*ddR') * op.gauss_rule.weights[iGauss]
            op.stiff[curNodes.+off, curNodes.+off] += 0.5 * Jac_ref_par * Jac_par_phys * op.EA * (du0dx + dw0dx^2) * (dR*dR') * op.gauss_rule.weights[iGauss]
            
            # only different entry of Jacobian
            op.jacob[curNodes.+off, curNodes.+off] += Jac_ref_par * Jac_par_phys * op.EA * (dw0dx^2) * (dR*dR') * op.gauss_rule.weights[iGauss]
            # check if the domain is correct
            domainLength += Jac_ref_par * Jac_par_phys * op.gauss_rule.weights[iGauss]
        end
        op.ext[curNodes] += localx
        op.ext[curNodes.+mesh.numBasis] += localy
    end
    # enforce symmetry K21 -> K12
    op.stiff[off+1:2off,1:off] .= op.stiff[1:off,off+1:2off]
    # form jacobian
    op.jacob .+= op.stiff
    
    # @show domainLength
    return nothing
end


function applyBC!(op::FEOperator)
    Neumann_BC = [op.BC[i] for i=eachindex(op.BC) if op.BC[i].type=="Neumann"]
    Dirichlet_BC = [op.BC[i] for i=eachindex(op.BC) if op.BC[i].type=="Dirichlet"]
    # apply Neumann BC
    Bmat = spzeros(Float64, length(Neumann_BC), size(op.stiff, 2))
    # rhs = spzeros(Float64, size(op.stiff,1)+length(Neumann_BC))
    for i=eachindex(Neumann_BC)
        off = (Neumann_BC[i].comp-1)*op.mesh.numBasis
        cdof_neu = findall(op.mesh.controlPoints[1,:].==Neumann_BC[i].x_val)
        for iElem=1:op.mesh.numElem
            curNodes = op.mesh.elemNode[iElem]
            if cdof_neu[1] in curNodes
                # nodes = gauss_rule.nodes
                nodes = [2*Neumann_BC[i].u_val-1] # map into parameter space
                # compute the Bernstein basis functions and derivatives
                B, dB, ddB = bernsteinBasis(nodes, op.mesh.degP[1])
                uMin = op.mesh.elemVertex[iElem, 1]
                uMax = op.mesh.elemVertex[iElem, 2]
                Jac_ref_par = (uMax-uMin)/2

                #compute the (B-)spline basis functions and derivatives with Bezier extraction
                N_mat = B * op.mesh.C[iElem]'
                dN_mat = dB * op.mesh.C[iElem]'/Jac_ref_par

                #compute the rational spline basis
                numNodes = length(curNodes)
                cpts = op.mesh.controlPoints[1, curNodes]
                wgts = op.mesh.weights[curNodes]
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

                    localLagrange += Jac_ref_par * Jac_par_phys * dR * op.gauss_rule.weights[iGauss]
                end
                Bmat[i,curNodes.+off] += localLagrange
                # lagrange multiplier entry
                global_resid[2mesh.numBasis+i] = Neumann_BC[i].op_val
            end
        end
    end
    op.stiff[end-size(Bmat,1)+1:end,1:size(Bmat,2)] .= Bmat
    op.stiff[1:size(Bmat,2),end-size(Bmat,1)+1:end] .= Bmat'
    op.jacob[end-size(Bmat,1)+1:end,1:size(Bmat,2)] .= Bmat
    op.jacob[1:size(Bmat,2),end-size(Bmat,1)+1:end] .= Bmat'
    # apply Dirichlet BC
    for i=eachindex(Dirichlet_BC)
        bcdof = Array{Int64,1}(undef, 0)
        bcdof = vcat(bcdof, findall(op.mesh.controlPoints[1,:].==Dirichlet_BC[i].x_val))
        for j ∈ Dirichlet_BC[i].comp
            bcdof .+= (j-1)*op.mesh.numBasis
            bcval = Array{Float64,1}(undef, 0)
            bcval = vcat(bcval, Dirichlet_BC[i].op_val)
            global_resid .-= op.stiff[:,bcdof]*bcval
            global_resid[bcdof] .= bcval
            op.stiff[bcdof, :] .= 0.0 
            op.stiff[:, bcdof] .= 0.0 
            op.jacob[bcdof,:] .= 0.0
            op.jacob[:,bcdof] .= 0.0
            op.stiff[bcdof, bcdof] = sparse(I, length(bcdof), length(bcdof))
            op.stiff[bcdof, bcdof] = sparse(I, length(bcdof), length(bcdof))
        end
    end
    return nothing
end