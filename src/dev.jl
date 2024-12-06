function applyBC!(op::AbstractFEOperator)
    numBasis = op.mesh.numBasis
    # apply Neumann BC
    for i=eachindex(op.Neumann_BC)
        off = (op.Neumann_BC[i].comp-1)*op.mesh.numBasis
        cdof_neu = findall(op.mesh.controlPoints[1,:].==op.Neumann_BC[i].x_val)
        for iElem=1:op.mesh.numElem
            # index of that element
            In = nodes(iElem,op.mesh.degP[1])
            if cdof_neu[1] in In
                # get the actual node position
                BC_nodes = [2*op.Neumann_BC[i].u_val-1] # map into parameter space
                # compute the Bernstein basis functions and derivatives
                B, dB, _ = bernsteinBasis(BC_nodes, op.mesh.degP[1])
                Jac_ref_par = Jξ(op.mesh.elemVertex,iElem)

                #compute the (B-)spline basis functions and derivatives with Bezier extraction
                N_mat = B * op.mesh.C[iElem]
                dN_mat = dB * op.mesh.C[iElem]/Jac_ref_par

                #compute the rational basis
                RR = N_mat[1,:].* op.mesh.weights[In]
                dR = dN_mat[1,:].* op.mesh.weights[In]
                w_sum = sum(RR)
                dw_xi = sum(dR)
                dR = dR/w_sum - RR*dw_xi/w_sum^2

                #compute the Jacobian of the transformation from parameter to physical space
                dxdxi = dR' * op.mesh.controlPoints[1,In]
                Jac_par_phys = det(dxdxi)

                # better indeexing
                op.stiff[2numBasis+i,In.+off] += Jac_ref_par * Jac_par_phys * dR * op.gauss_rule.weights[1]
                op.stiff[In.+off,2numBasis+i] += Jac_ref_par * Jac_par_phys * dR * op.gauss_rule.weights[1]

                # lagrange multiplier entry
                op.resid[2numBasis+i] = op.Neumann_BC[i].op_val
            end
        end
    end
    # propagate the Neumann BC to the jacobian
    op.jacob[2numBasis+1:end,:] .= op.stiff[2numBasis+1:end,:]
    op.jacob[:,2numBasis+1:end] .= op.stiff[:,2numBasis+1:end]
    
    # apply Dirichlet BC
    for i=eachindex(op.Dirichlet_BC)
        bcdof = findall(op.mesh.controlPoints[1,:].==op.Dirichlet_BC[i].x_val)
        for j ∈ op.Dirichlet_BC[i].comp
            # get the actual node position
            bcdof .+= (j-1)*op.mesh.numBasis
            # get the BC value
            bcval = op.Dirichlet_BC[i].op_val
            op.resid .-= op.stiff[:,bcdof]*bcval
            op.resid[bcdof] .= bcval
            # reset stiffness and mass
            op.stiff[bcdof,:] .= op.jacob[bcdof,:] .= 0.0 
            op.stiff[:,bcdof] .= op.jacob[:,bcdof] .= 0.0
            op.stiff[bcdof,bcdof] .= op.jacob[bcdof,bcdof] .= 1.0
        end
    end
end


# function integrate_fast!(op::AbstractFEOperator, x0::AbstractVector{T}, fx::AbstractArray{T}) where T
#     # reset
#     op.stiff .= 0.; op.jacob .= 0.; op.ext .= 0.;
#     degP = op.mesh.degP[1]; off = op.mesh.numBasis
#     # integrate on each element
#     for iElem ∈ 1:Nelem

#         # compute the (B-)spline basis functions and derivatives with Bezier extraction
#         Jξ,N,dN,ddN = BSplineBasis(op.mesh, iElem)
        
#         # integrate first term
#         @integrate op.stiff[I] += op.EA(phys_pt(I))*(dR(nodes,N,dN,w,iElem,degP)*dR(nodes,N,dN,w,iElem,degP)) over I in element(iElem,degP)
#         @integrate op.stiff[I+δ(off)] += op.EA(phys_pt) * (dw0dx) * (dR*dR') 
#         @integrate op.stiff[I+δ(off,off)] += op.EI(phys_pt) * (ddR*ddR') * op.gauss_rule.weights[iGauss]
#         @integrate op.stiff[I+δ(off,off)] += op.EA(phys_pt) * (du0dx + dw0dx^2) * (dR*dR') * op.gauss_rule.weights[iGauss]
#     end
# end


# function phys_pt(op,nodes,N,dN,w,iElem,degP)
#     I = CartesianIndex(iElem:iElem+degP)
#     for iGauss = 1:length(nodes)
#         #compute the rational basis
#         RR = N[iGauss,:] .* w[I]
#         dR = dN[iGauss,:] .* w[I]
#         w_sum = sum(RR)
#         dw_xi = sum(dR)
#         dR = dR/w_sum - RR*dw_xi/w_sum^2
#         phys_pt = RR' * op.mesh.controlPoints[1, curNodes]
#     end
#     return phys_pt
# end

# function dR(nodes,N,dN,w,iElem,degP) where N
#     I = CartesianIndex(iElem:iElem+degP)
#     for iGauss = 1:length(nodes)
#         #compute the rational basis
#         RR = N[iGauss,:] .* w[I]
#         dR = dN[iGauss,:] .* w[I]
#         w_sum = sum(RR)
#         dw_xi = sum(dR)
#         dR = dR/w_sum - RR*dw_xi/w_sum^2
#     end
#     return dR[I]
# end

@fastmath @inline function mult(I::CartesianIndex,A,x)
    s = zero(eltype(A))
    for J in CartesianIndices(x)
        s += @inbounds(x[J]*A[I,J])
    end
    return s
end
