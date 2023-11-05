using SparseArrays
using UnPack

# Abstract FEA operaotr
abstract type AbstractFEOperator end

struct EulerBeam <: AbstractFEOperator
    x :: Vector{Float64}
    resid :: Vector{Float64}
    jacob :: Matrix{Float64}
    EI    :: Float64
    EA :: Float64
    f :: Function
    mesh :: Mesh
    gauss_rule :: GaussQuad
    Dirichlet_BC :: Vector{Boundary1D}
    Neumann_BC :: Vector{Boundary1D}
    function EulerBeam(EI, EA, f, mesh, gauss_rule, Dirichlet_BC=[], Neumann_BC=[])
        numNeuBC = length(Neumann_BC)
        # generate
        x = zeros(Float64, 2*mesh.numBasis+numNeuBC)
        resid = zeros(Float64, 2*mesh.numBasis+numNeuBC)
        jacob = spzeros(2*mesh.numBasis+numNeuBC, 2*mesh.numBasis+numNeuBC)
        rhs = zeros(Float64, 2*mesh.numBasis+numNeuBC)
        new(x, resid, jacob, EI, EA, f, mesh, gauss_rule, Dirichlet_BC, Neumann_BC)
    end
end 

# function dynamic_update!(resid, jac, mass, xᵏ, aᵏ, problem, α₁, α₂, β, Δt)
#     # allocate stuff
#     @unpack jacob = problem
#     stiff = spzeros(size(jacob))
#     rhs = spzeros(size(resid))

#     # update stiffness and jacobian
#     update_global!(stiff, jacob, xᵏ, problem.mesh, problem.gauss_rule, problem)

#     # update rhs vector
#     update_external!(rhs, problem.mesh, problem.f, problem.gauss_rule)

#     # apply boundary consitions
#     jac .= α₁/(β*Δt^2)*mass + α₂*jacob
#     applyBCGlobal!(jac, stiff, rhs, problem.mesh,
#                    problem.Dirichlet_BC, problem.Neumann_BC,
#                    problem.gauss_rule)

#     # compute resdiuals
#     resid .= mass*aᵏ + stiff*xᵏ - rhs
#     return nothing
# end

# function dynamic_residuals!(resid, x₀, a₀ , mass, problem)

#     # allocate stuff
#     @unpack jacob = problem
#     stiff = spzeros(size(jacob))
#     rhs = spzeros(size(resid))

#     # update stiffness and jacobian
#     update_global!(stiff, jacob, x₀, problem.mesh, problem.gauss_rule, problem)

#     # update rhs vector
#     update_external!(rhs, problem.mesh, problem.f, problem.gauss_rule)

#     # apply boundary consitions
#     applyBCGlobal!(stiff, jacob, rhs, problem.mesh,
#                    problem.Dirichlet_BC, problem.Neumann_BC,
#                    problem.gauss_rule)

#     # compute resdiuals
#     resid .= mass*a₀ + stiff*x₀ - rhs
#     return nothing
# end


# function dynamic_jacobian!(jac, x₀, α₁, α₂, β, Δt, mass, problem)
     
#     # allocate stuff
#      @unpack jacob = problem
#      stiff = spzeros(size(jacob))
#      resid = spzeros(size(x₀))
 
#      # extract component
#      update_global!(stiff, jacob, x₀, problem.mesh, problem.gauss_rule, problem)
 
#      # apply boundary consitions
#      jacob .= α₁/(β*Δt^2)*mass + α₂*jacob
#      applyBCGlobal!(stiff, jacob, resid, problem.mesh,
#                     problem.Dirichlet_BC, problem.Neumann_BC,
#                     problem.gauss_rule)
 
#      # avoid nul effect of function as jacob is defined with the function           
#     jac .= jacob #α₁/(β*Δt^2)*M +  α₂*jacob
#     return nothing
# end


# function static_residuals!(resid, x₀, problem)

#     # allocate stuff
#     @unpack jacob = problem
#     stiff = spzeros(size(jacob))
#     rhs = spzeros(size(resid))

#     # update stiffness and jacobian
#     update_global!(stiff, jacob, x₀, problem.mesh, problem.gauss_rule, problem)

#     # update rhs vector
#     update_external!(rhs, problem.mesh, problem.f, problem.gauss_rule)

#     # apply boundary consitions
#     applyBCGlobal!(stiff, jacob, rhs, problem.mesh,
#                    problem.Dirichlet_BC, problem.Neumann_BC,
#                    problem.gauss_rule)

#     # compute resdiuals
#     resid .= stiff*x₀ - rhs

#     return nothing
# end


# function static_jacobian!(jac, x₀, problem)

#     # allocate stuff
#     @unpack jacob = problem
#     stiff = spzeros(size(jacob))
#     resid = spzeros(size(x₀))

#     # extract component
#     update_global!(stiff, jacob, x₀, problem.mesh, problem.gauss_rule, problem)

#     # apply boundary consitions
#     applyBCGlobal!(stiff, jacob, resid, problem.mesh,
#                    problem.Dirichlet_BC, problem.Neumann_BC,
#                    problem.gauss_rule)

#     # avoid nul effect of function as jacob is defined with the function           
#     jac .= jacob
#     return nothing
# end



function applyBCGlobal!(global_stiff, global_jacob, global_resid, mesh, Dirichlet_BC, Neumann_BC, gauss_rule)
    # apply Neumann BC
    Bmat = spzeros(Float64, length(Neumann_BC), size(global_stiff, 2))
    # rhs = spzeros(Float64, size(global_stiff,1)+length(Neumann_BC))
    for i=eachindex(Neumann_BC)
        off = (Neumann_BC[i].comp-1)*mesh.numBasis
        cdof_neu = findall(mesh.controlPoints[1,:].==Neumann_BC[i].x_val)
        for iElem=1:mesh.numElem
            curNodes = mesh.elemNode[iElem]
            if cdof_neu[1] in curNodes
                # nodes = gauss_rule.nodes
                nodes = [2*Neumann_BC[i].u_val-1] # map into parameter space
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
                Bmat[i,curNodes.+off] += localLagrange
                # lagrange multiplier entry
                global_resid[2mesh.numBasis+i] = Neumann_BC[i].op_val
            end
        end
    end
    global_stiff[end-size(Bmat,1)+1:end,1:size(Bmat,2)] .= Bmat
    global_stiff[1:size(Bmat,2),end-size(Bmat,1)+1:end] .= Bmat'
    global_jacob[end-size(Bmat,1)+1:end,1:size(Bmat,2)] .= Bmat
    global_jacob[1:size(Bmat,2),end-size(Bmat,1)+1:end] .= Bmat'
    # apply Dirichlet BC
    for i=eachindex(Dirichlet_BC)
        bcdof = Array{Int64,1}(undef, 0)
        bcdof = vcat(bcdof, findall(mesh.controlPoints[1,:].==Dirichlet_BC[i].x_val))
        for j ∈ Dirichlet_BC[i].comp
            bcdof .+= (j-1)*mesh.numBasis
            bcval = Array{Float64,1}(undef, 0)
            bcval = vcat(bcval, Dirichlet_BC[i].op_val)
            global_resid .-= global_stiff[:,bcdof]*bcval
            global_resid[bcdof] .= bcval
            global_stiff[bcdof, :] .= 0.0 
            global_stiff[:, bcdof] .= 0.0 
            global_jacob[bcdof,:] .= 0.0
            global_jacob[:,bcdof] .= 0.0
            global_stiff[bcdof, bcdof] = sparse(I, length(bcdof), length(bcdof))
            global_jacob[bcdof, bcdof] = sparse(I, length(bcdof), length(bcdof))
        end
    end
    return nothing
end


function interate!(global_stiff, global_jacob, external, x0, problem)
    off = problem.mesh.numBasis
    global_stiff[1:2off,1:2off] .= 0.;
    global_jacob[1:2off,1:2off] .= 0.;
    B, dB, ddB = bernsteinBasis(problem.gauss_rule.nodes, problem.mesh.degP[1])
    domainLength = 0
    for iElem = 1:problem.mesh.numElem
        uMin = problem.mesh.elemVertex[iElem, 1]
        uMax = problem.mesh.elemVertex[iElem, 2]
        Jac_ref_par = (uMax-uMin)/2

        #compute the (B-)spline basis functions and derivatives with Bezier extraction
        N_mat = B * problem.mesh.C[iElem]'
        dN_mat = dB * problem.mesh.C[iElem]'/Jac_ref_par
        ddN_mat = ddB * problem.mesh.C[iElem]'/Jac_ref_par^2

        #compute the rational spline basis
        curNodes = problem.mesh.elemNode[iElem]
        cpts = problem.mesh.controlPoints[1, curNodes]
        wgts = problem.mesh.weights[curNodes]

        # integrate on element
        for iGauss = 1:length(problem.gauss_rule.nodes)
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

            # compute the different terms
            global_stiff[curNodes, curNodes] += Jac_ref_par * Jac_par_phys * problem.EA * (dR*dR') * problem.gauss_rule.weights[iGauss]
            global_stiff[curNodes, curNodes.+off] += 0.5 * Jac_ref_par * Jac_par_phys * problem.EA * (dw0dx) * (dR*dR') * problem.gauss_rule.weights[iGauss]
            global_stiff[curNodes.+off, curNodes.+off] += Jac_ref_par * Jac_par_phys^2 * problem.EI * (ddR*ddR') * problem.gauss_rule.weights[iGauss]
            global_stiff[curNodes.+off, curNodes.+off] += 0.5 * Jac_ref_par * Jac_par_phys * problem.EA * (du0dx + dw0dx^2) * (dR*dR') * problem.gauss_rule.weights[iGauss]
            
            # only different entry of Jacobian
            global_jacob[curNodes.+off, curNodes.+off] += Jac_ref_par * Jac_par_phys * problem.EA * (dw0dx^2) * (dR*dR') * problem.gauss_rule.weights[iGauss]
            # check if the domain is correct
            domainLength += Jac_ref_par * Jac_par_phys * problem.gauss_rule.weights[iGauss]
        end
    end
    # enforce symmetry K21 -> K12
    global_stiff[off+1:2off,1:off] .= global_stiff[1:off,off+1:2off]
    # form jacobian
    global_jacob .+= global_stiff
    # @show domainLength
    return nothing
end

function update_global!(global_stiff, global_jacob, x0, mesh, gauss_rule, problem)
    off = mesh.numBasis
    global_stiff[1:2off,1:2off] .= 0.;
    global_jacob[1:2off,1:2off] .= 0.;
    B, dB, ddB = bernsteinBasis(gauss_rule.nodes, mesh.degP[1])
    domainLength = 0
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
        cpts = mesh.controlPoints[1, curNodes]
        wgts = mesh.weights[curNodes]

        # numNodes = length(curNodes)
        # localx = zeros(numNodes)
        # localy = zeros(numNodes)

        # integrate on element
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
            du0dx = dR' * x0[curNodes]
            dw0dx = dR' * x0[curNodes.+off]

            # compute the different terms
            global_stiff[curNodes, curNodes] += Jac_ref_par * Jac_par_phys * problem.EA * (dR*dR') * gauss_rule.weights[iGauss]
            global_stiff[curNodes, curNodes.+off] += 0.5 * Jac_ref_par * Jac_par_phys * problem.EA * (dw0dx) * (dR*dR') * gauss_rule.weights[iGauss]
            global_stiff[curNodes.+off, curNodes.+off] += Jac_ref_par * Jac_par_phys^2 * problem.EI * (ddR*ddR') * gauss_rule.weights[iGauss]
            global_stiff[curNodes.+off, curNodes.+off] += 0.5 * Jac_ref_par * Jac_par_phys * problem.EA * (du0dx + dw0dx^2) * (dR*dR') * gauss_rule.weights[iGauss]
            
            # external force at physical point
            # fi = fx[:,(iElem-1)*problem.mesh.numElem+iGauss]
            # localx += Jac_par_phys * Jac_ref_par * fi[1] * RR * problem.gauss_rule.weights[iGauss]
            # localy += Jac_par_phys * Jac_ref_par * fi[2] * RR * problem.gauss_rule.weights[iGauss]
            
            # only different entry of Jacobian
            global_jacob[curNodes.+off, curNodes.+off] += Jac_ref_par * Jac_par_phys * problem.EA * (dw0dx^2) * (dR*dR') * gauss_rule.weights[iGauss]
            # check if the domain is correct
            domainLength += Jac_ref_par * Jac_par_phys * gauss_rule.weights[iGauss]
        end
    end
    # external[curNodes] += localx
    # external[curNodes.+problem.mesh.numBasis] += localy
    # enforce symmetry K21 -> K12
    global_stiff[off+1:2off,1:off] .= global_stiff[1:off,off+1:2off]
    # form jacobian
    global_jacob .+= global_stiff
    # @show domainLength
    return nothing
end


"""
Assembles the RHS vector corresponding to the body force
RHS[i]=∫_Ω ϕ_i(x)*f(x) dΩ
"""
function update_external!(rhs, mesh::Mesh, fx::Function, gauss_rule)
    B, dB, ddB = bernsteinBasis(gauss_rule.nodes, mesh.degP[1])
    domainLength = 0; rhs .= 0.0
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
        localx = zeros(numNodes)
        localy = zeros(numNodes)
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

            # external force at physical point
            phys_pt = RR'*cpts
            fi = fx(phys_pt)
            localx += Jac_par_phys * Jac_ref_par * fi[1] * RR * gauss_rule.weights[iGauss]
            localy += Jac_par_phys * Jac_ref_par * fi[2] * RR * gauss_rule.weights[iGauss]
            domainLength += Jac_par_phys * Jac_ref_par * gauss_rule.weights[iGauss]
        end
        rhs[curNodes] += localx
        rhs[curNodes.+mesh.numBasis] += localy
    end
    # @show domainLength
    return nothing
end
function update_external!(rhs, mesh::Mesh, fx::Matrix, gauss_rule)
    B, dB, ddB = bernsteinBasis(gauss_rule.nodes, mesh.degP[1])
    domainLength = 0; rhs .= 0.0
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
        localx = zeros(numNodes)
        localy = zeros(numNodes)
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

            # external force at physical point
            fi = fx[:,(iElem-1)*mesh.numElem+iGauss]
            localx += Jac_par_phys * Jac_ref_par * fi[1] * RR * gauss_rule.weights[iGauss]
            localy += Jac_par_phys * Jac_ref_par * fi[2] * RR * gauss_rule.weights[iGauss]
            domainLength += Jac_par_phys * Jac_ref_par * gauss_rule.weights[iGauss]
        end
        rhs[curNodes] += localx
        rhs[curNodes.+mesh.numBasis] += localy
    end
    # @show domainLength
    return nothing
end
 
function global_mass!(mass, mesh::Mesh, ρ, gauss_rule)
    B, dB = bernsteinBasis(gauss_rule.nodes, mesh.degP[1])
    domainLength = 0
    # mass = spzeros(Float64, mesh.numBasis, mesh.numBasis)
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
            phys_pt = RR' * cpts
            localMass += Jac_par_phys * Jac_ref_par * ρ(phys_pt) * (RR*RR') * gauss_rule.weights[iGauss]
            domainLength += Jac_par_phys * Jac_ref_par * gauss_rule.weights[iGauss]
        end
        #@show localMass
        #readline(stdin)
        mass[curNodes, curNodes] += localMass
        mass[curNodes.+mesh.numBasis, curNodes.+mesh.numBasis] += localMass
    end
    # @show domainLength
    return mass
end



# ## old functions
# """
#     Assembles the stiffness matrix for 2D Elasticity Kₑ = ∫ B^T*C*B dΩ
# """
# function assemble_stiff(mesh::Mesh, EI, EA, gauss_rule)
#     w0 = zeros(Float64, mesh.numBasis)
#     return assemble_stiff(mesh, w0, w0, EI, EA, gauss_rule)
# end

# """
# """
# function assemble_stiff(mesh::Mesh, u0, w0, EI, EA, gauss_rule)
#     B, dB, ddB = bernsteinBasis(gauss_rule.nodes, mesh.degP[1])
#     domainLength = 0
#     K22 = spzeros(mesh.numBasis, mesh.numBasis)
#     K11 = spzeros(mesh.numBasis, mesh.numBasis)
#     K12 = spzeros(mesh.numBasis, mesh.numBasis)
#     J22 = spzeros(mesh.numBasis, mesh.numBasis)
#     for iElem = 1:mesh.numElem
#         uMin = mesh.elemVertex[iElem, 1]
#         uMax = mesh.elemVertex[iElem, 2]
#         Jac_ref_par = (uMax-uMin)/2

#         #compute the (B-)spline basis functions and derivatives with Bezier extraction
#         N_mat = B * mesh.C[iElem]'
#         dN_mat = dB * mesh.C[iElem]'/Jac_ref_par
#         ddN_mat = ddB * mesh.C[iElem]'/Jac_ref_par^2

#         #compute the rational spline basis
#         curNodes = mesh.elemNode[iElem]
#         # numNodes = length(curNodes)
#         cpts = mesh.controlPoints[1, curNodes]
#         wgts = mesh.weights[curNodes]
#         # k11 = zeros(numNodes, numNodes)
#         # k12 = zeros(numNodes, numNodes)
#         # k22 = zeros(numNodes, numNodes)
#         # j22 = zeros(numNodes, numNodes)
#         for iGauss = 1:length(gauss_rule.nodes)
#             #compute the rational basis
#             RR = N_mat[iGauss,:].* wgts
#             dR = dN_mat[iGauss,:].* wgts
#             ddR = ddN_mat[iGauss,:].* wgts
#             w_sum = sum(RR)
#             dw_xi = sum(dR)
#             dR = dR/w_sum - RR*dw_xi/w_sum^2
#             ddR = ddR/w_sum - 2*dR*dw_xi/w_sum^2 - RR*sum(ddR)/w_sum^2 + 2*RR*dw_xi^2/w_sum^3

#             #compute the derivatives w.r.t to the physical space
#             dxdxi = dR' * cpts
#             Jac_par_phys = det(dxdxi)

#             # compute linearised terms using the current solution
#             du0dx = dR' * u0[curNodes]
#             dw0dx = dR' * w0[curNodes]

#             # compute the different terms
#             # k11 += Jac_ref_par * Jac_par_phys * EA * (dR*dR') * gauss_rule.weights[iGauss]
#             # k12 += 0.5 * Jac_ref_par * Jac_par_phys * EA * (dw0dx) * (dR*dR') * gauss_rule.weights[iGauss]
#             # k22 += Jac_ref_par * Jac_par_phys^2 * EI * (ddR*ddR') * gauss_rule.weights[iGauss]
#             # k22 += 0.5 * Jac_ref_par * Jac_par_phys * EA * (du0dx + dw0dx^2) * (dR*dR') * gauss_rule.weights[iGauss]
#             K11[curNodes, curNodes] += Jac_ref_par * Jac_par_phys * EA * (dR*dR') * gauss_rule.weights[iGauss]
#             K12[curNodes, curNodes] += 0.5 * Jac_ref_par * Jac_par_phys * EA * (dw0dx) * (dR*dR') * gauss_rule.weights[iGauss]
#             K22[curNodes, curNodes] += Jac_ref_par * Jac_par_phys^2 * EI * (ddR*ddR') * gauss_rule.weights[iGauss]
#             K22[curNodes, curNodes] += 0.5 * Jac_ref_par * Jac_par_phys * EA * (du0dx + dw0dx^2) * (dR*dR') * gauss_rule.weights[iGauss]
            
#             # only different entry of Jacobian
#             J22[curNodes, curNodes] += Jac_ref_par * Jac_par_phys * EA* (dw0dx^2) * (dR*dR') * gauss_rule.weights[iGauss]
#             # check if the domain is correct
#             domainLength += Jac_ref_par * Jac_par_phys * gauss_rule.weights[iGauss]
#         end
#         # K11[curNodes, curNodes] += k11
#         # K12[curNodes, curNodes] += k12
#         # K22[curNodes, curNodes] += k22
#         # J22[curNodes, curNodes] += j22
#     end
#     # @show domainLength
#     # symmetrized case K21 = K12
#     return K11,K12,K22,J22
# end



# """
# Assembles the RHS vector corresponding to the body force
# RHS[i]=∫_Ω ϕ_i(x)*f(x) dΩ
# """
# function assemble_rhs(mesh::Mesh, f::Function, gauss_rule)
#     B, dB, ddB = bernsteinBasis(gauss_rule.nodes, mesh.degP[1])
#     domainLength = 0
#     rhs = zeros(Float64, mesh.numBasis)
#     for iElem = 1:mesh.numElem
#         uMin = mesh.elemVertex[iElem, 1]
#         uMax = mesh.elemVertex[iElem, 2]
#         Jac_ref_par = (uMax-uMin)/2
#         #@show Jac_ref_par
#         #compute the (B-)spline basis functions and derivatives with Bezier extraction
#         N_mat = B * mesh.C[iElem]'
#         dN_mat = dB * mesh.C[iElem]'/Jac_ref_par

#         #compute the rational spline basis
#         curNodes = mesh.elemNode[iElem]
#         numNodes = length(curNodes)
#         cpts = mesh.controlPoints[1, curNodes]
#         wgts = mesh.weights[curNodes]
#         localRhs = zeros(numNodes)
#         for iGauss = 1:length(gauss_rule.nodes)
#             #compute the rational basis
#             RR = N_mat[iGauss,:].* wgts
#             dR = dN_mat[iGauss,:].* wgts
#             w_sum = sum(RR)
#             dw_xi = sum(dR)
#             dR = dR/w_sum - RR*dw_xi/w_sum^2

#             #compute the derivatives w.r.t to the physical space
#             dxdxi = dR' * cpts
#             Jac_par_phys = det(dxdxi)
#             RR /= w_sum

#             phys_pt = RR'*cpts
#             localRhs += Jac_par_phys * Jac_ref_par * f(phys_pt) * RR * gauss_rule.weights[iGauss]
#             domainLength += Jac_par_phys * Jac_ref_par * gauss_rule.weights[iGauss]
#         end
#         rhs[curNodes] += localRhs
#     end
#     # @show domainLength
#     return rhs
# end


# function assemble(K11, K12, K22, r11, r22)
#     K = spzeros(size(K11,1)+size(K22,1), size(K11,2)+size(K22,2))
#     R = zeros(size(r11,1)+size(r22,1))
#     K[1:size(K11,1),1:size(K11,2)] .= K11
#     K[end-size(K22,1)+1:end,end-size(K22,2)+1:end] .= K22
#     K[1:size(K12,1),size(K11,2)+1:size(K11,2)+size(K12,2)] .= K12
#     K[size(K11,1)+1:size(K11,1)+size(K12,1),1:size(K12,2)] .= K12'
#     R[1:size(r11,1)] .= r11
#     R[end-size(r22,1)+1:end] .= r22
#     return K,R
# end

# function Jacobian(K11,K12,K22,J22)
#     J = spzeros(size(K11,1)+size(K22,1), size(K11,2)+size(K22,2))
#     J[1:size(K11,1),1:size(K11,2)] .= K11
#     J[1:size(K12,1),size(K11,2)+1:size(K11,2)+size(K12,2)] .= K12
#     J[size(K11,1)+1:size(K11,1)+size(K12,1),1:size(K12,2)] .= K12'
#     J[end-size(K22,1)+1:end,end-size(K22,2)+1:end] .= K22
#     dx=size(K22,1)-size(J22,1); dy=size(K22,2)-size(J22,2)
#     J[end-size(K22,1)+1:end-dx,end-size(K22,2)+1:end-dy] .+= J22
#     return J
# end



# function applyBCDirichlet(lhs, rhs, bound_cond, mesh)
#     for i=eachindex(bound_cond)
#         bcdof = Array{Int64,1}(undef, 0)
#         bcval = Array{Float64,1}(undef, 0)
#         bcdof = vcat(bcdof, findall(mesh.controlPoints[1,:].==bound_cond[i].x_val))
#         bcval = vcat(bcval, bound_cond[i].op_val)
#         rhs = rhs - lhs[:,bcdof]*bcval
#         rhs[bcdof] = bcval
#         lhs[bcdof, :] .= 0.
#         lhs[:, bcdof] .= 0.
#         lhs[bcdof, bcdof] = sparse(I, length(bcdof), length(bcdof))
#     end
#     return lhs, rhs
# end

# # strikes trough the LHS where the BC matches the index
# function applyBCDirichlet(lhs, bound_cond, mesh)
#     for i=eachindex(bound_cond)
#         bcdof = Array{Int64,1}(undef, 0)
#         bcdof = vcat(bcdof, findall(mesh.controlPoints[1,:].==bound_cond[i].x_val))
#         lhs[bcdof, :] .= 0.
#         lhs[:, bcdof] .= 0.
#     end
#     return lhs
# end

# function applyBCNeumann(stiff, f, bound_cond, mesh, gauss_rule)
#     Bmat = spzeros(Float64, length(bound_cond), mesh.numBasis)
#     rhs = zeros(Float64, size(stiff,1)+length(bound_cond))
#     for i=eachindex(bound_cond)
#         cdof_neu = findall(mesh.controlPoints[1,:].==bound_cond[i].x_val)
#         for iElem=1:mesh.numElem
#             curNodes = mesh.elemNode[iElem]
#             if cdof_neu[1] in curNodes
#                 # nodes = gauss_rule.nodes
#                 nodes = [2*bound_cond[i].u_val-1] # map into parameter space
#                 # compute the Bernstein basis functions and derivatives
#                 B, dB, ddB = bernsteinBasis(nodes, mesh.degP[1])
#                 uMin = mesh.elemVertex[iElem, 1]
#                 uMax = mesh.elemVertex[iElem, 2]
#                 Jac_ref_par = (uMax-uMin)/2

#                 #compute the (B-)spline basis functions and derivatives with Bezier extraction
#                 N_mat = B * mesh.C[iElem]'
#                 dN_mat = dB * mesh.C[iElem]'/Jac_ref_par

#                 #compute the rational spline basis
#                 numNodes = length(curNodes)
#                 cpts = mesh.controlPoints[1, curNodes]
#                 wgts = mesh.weights[curNodes]
#                 localLagrange = zeros(numNodes)
#                 for iGauss = 1:length(nodes)
#                     #compute the rational basis
#                     RR = N_mat[iGauss,:].* wgts
#                     dR = dN_mat[iGauss,:].* wgts
#                     w_sum = sum(RR)
#                     dw_xi = sum(dR)
#                     dR = dR/w_sum - RR*dw_xi/w_sum^2

#                     #compute the Jacobian of the transformation from parameter to physical space
#                     dxdxi = dR' * cpts
#                     Jac_par_phys = det(dxdxi)

#                     localLagrange += Jac_ref_par * Jac_par_phys * dR * gauss_rule.weights[iGauss]
#                 end
#                 Bmat[i,curNodes] += localLagrange
#                 rhs[mesh.numBasis+i] = bound_cond[i].op_val
#             end
#         end
#     end
#     rhs[1:size(stiff,1)] .= f
#     lhs = spzeros(size(stiff,1)+length(bound_cond), size(stiff,2)+length(bound_cond))
#     lhs[1:size(stiff,1),1:size(stiff,2)] .= sparse(stiff)
#     lhs[end-size(Bmat,1)+1:end,1:size(Bmat,2)] .= Bmat
#     lhs[1:size(Bmat,2),end-size(Bmat,1)+1:end] .= Bmat'
#     return lhs, rhs
# end

