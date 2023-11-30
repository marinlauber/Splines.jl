using SparseArrays
using UnPack

# Abstract FEA operaotr
abstract type AbstractFEOperator end

"""
    FEOperator

Defines a finite element operator for a beam element, with the following fields:
    x :: Vector{Float64} : current solution
    resid :: Vector{Float64} : residual
    stiff :: Matrix{Float64} : stiffness matrix
    jacob :: Matrix{Float64} : jacobian matrix
    ext   :: Vector{Float64} : external force
    mass :: Union{Float64, Matrix{Float64}} : mass matrix
    mesh :: Mesh : mesh
    gauss_rule :: GaussQuad : gauss quadrature rule
    Dirichlet_BC :: Vector{Boundary1D} : Dirichlet boundary conditions
    Neumann_BC :: Vector{Boundary1D} : Neumann boundary conditions
    EI :: Float64 : bending stiffness
    EA :: Float64 : axial stiffness
"""
struct FEOperator <: AbstractFEOperator
    x :: Vector{Float64}
    resid :: Vector{Float64}
    stiff :: Matrix{Float64}
    jacob :: Matrix{Float64}
    ext   :: Vector{Float64}
    mass :: Union{Float64, Matrix{Float64}}
    mesh :: Mesh
    gauss_rule :: GaussQuad
    Dirichlet_BC :: Vector{Boundary1D}
    Neumann_BC :: Vector{Boundary1D}
    EI :: Float64
    EA :: Float64
    function FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC=[], Neumann_BC=[]; ρ::Function=(x)->-1.0)
        numNeuBC = length(Neumann_BC)
        # generate
        x = zeros(Float64, 2*mesh.numBasis+numNeuBC)
        resid = zeros(Float64, 2*mesh.numBasis+numNeuBC)
        ext = zeros(Float64, 2*mesh.numBasis+numNeuBC)
        stiff = spzeros(2*mesh.numBasis+numNeuBC, 2*mesh.numBasis+numNeuBC)
        jacob = spzeros(2*mesh.numBasis+numNeuBC, 2*mesh.numBasis+numNeuBC)
        mass = zeros(2*mesh.numBasis+numNeuBC, 2*mesh.numBasis+numNeuBC)
        # no mass if not provided
        ρ(0.5)>0.0 ? global_mass!(mass, mesh, ρ, gauss_rule) : mass = 0.0
        new(x, resid, stiff, jacob, ext, mass, mesh, gauss_rule, Dirichlet_BC, Neumann_BC, EI, EA)
    end
end
# backward compatibiity
EulerBeam(EI, EA, f, mesh, gr, D_BC=[], N_BC=[]) = FEOperator(mesh, gr, EI, EA, D_BC, N_BC)

"""
    residual(resid, x, force, op::FEOperator)

Compute the residuals given a operator `op`, an initial solution `x` and an 
external force `force`.
"""
function residual!(resid, x, force, op::FEOperator)
    # update the jacobian, the residual and the external force
    integrate!(op, x, force)

    # apply BC
    applyBC!(op)

    # assign residual
    resid .= op.stiff*x - op.ext
    return nothing
end


"""
    jacobian!(jacob, x, force, op::FEOperator)

Compute the jacobia given a operator `op`, an initial solution `x` and an 
external force `force`.
"""
function jacobian!(jacob, x, force, op::FEOperator)
    # update the jacobian, the residual and the external force
    integrate!(op, x, force)

    # apply BC
    applyBC!(op)

    # assign jacobian
    jacob .= op.jacob
    return nothing
end

"""
    integrate!(op::FEOperator, x0::Vector{Float64}, fx)

Integrate the operator `op` given an initial solution `x0` and an external force. 
This performs the element-wise integration og the different term in the governing equations
"""
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  29.019 μs …  2.566 ms  ┊ GC (min … max):  0.00% … 97.31%
#  Time  (median):     30.990 μs              ┊ GC (median):     0.00%
#  Time  (mean ± σ):   37.714 μs ± 87.849 μs  ┊ GC (mean ± σ):  12.85% ±  5.44%

#   ▃▆█▇▅▃▂▃▃▃▂▁                                                ▂
#   █████████████▇▇▆▅▆▇▆▅▅▆▅▅▄▄▄▆▆▆▆▆▅▄▅▆▇█▇▆▆▆▆▅▄▅▇█▇▆▅▆▁▅▆██▇ █
#   29 μs        Histogram: log(frequency) by time      68.9 μs <

#  Memory estimate: 150.39 KiB, allocs estimate: 1160.
function integrate!(op::FEOperator, x0::Vector{Float64}, fx)
    off = op.mesh.numBasis

    # reset
    op.stiff .= 0.; op.jacob .= 0.; op.ext .= 0.0;

    # precompute bernstein basis
    B, dB, ddB = bernsteinBasis(op.gauss_rule.nodes, op.mesh.degP[1])
    numGauss = length(op.gauss_rule.nodes)
    domainLength = 0
    for iElem = 1:op.mesh.numElem
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

            # external force at physical point fx{size{2,numElem*numGauss}]}
            fi = fx[:,(iElem-1)*numGauss+iGauss]
            op.ext[curNodes     ] += Jac_par_phys * Jac_ref_par * fi[1] * RR * op.gauss_rule.weights[iGauss]
            op.ext[curNodes.+off] += Jac_par_phys * Jac_ref_par * fi[2] * RR * op.gauss_rule.weights[iGauss]

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
    end
    # enforce symmetry K21 -> K12
    op.stiff[off+1:2off,1:off] .= op.stiff[1:off,off+1:2off]
    # form jacobian
    op.jacob .+= op.stiff
    
    # @show domainLength
    return nothing
end

# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  11.909 μs …  4.296 ms  ┊ GC (min … max):  0.00% … 98.00%
#  Time  (median):     12.763 μs              ┊ GC (median):     0.00%
#  Time  (mean ± σ):   15.591 μs ± 93.350 μs  ┊ GC (mean ± σ):  13.21% ±  2.20%

#     █▇▂                                                        
#   ▃████▆▄▃▃▃▄▄▄▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▂ ▃
#   11.9 μs         Histogram: frequency by time        24.5 μs <

#  Memory estimate: 25.81 KiB, allocs estimate: 421.
function applyBC!(op::FEOperator)
    # apply Neumann BC
    Bmat = spzeros(Float64, length(op.Neumann_BC), size(op.stiff, 2))
    for i=eachindex(op.Neumann_BC)
        off = (op.Neumann_BC[i].comp-1)*op.mesh.numBasis
        cdof_neu = findall(op.mesh.controlPoints[1,:].==op.Neumann_BC[i].x_val)
        for iElem=1:op.mesh.numElem
            curNodes = op.mesh.elemNode[iElem]
            if cdof_neu[1] in curNodes
                # nodes = gauss_rule.nodes
                nodes = [2*op.Neumann_BC[i].u_val-1] # map into parameter space
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
                op.resid[2op.mesh.numBasis+i] = op.Neumann_BC[i].op_val
            end
        end
    end
    op.stiff[end-size(Bmat,1)+1:end,1:size(Bmat,2)] .= Bmat
    op.stiff[1:size(Bmat,2),end-size(Bmat,1)+1:end] .= Bmat'
    op.jacob[end-size(Bmat,1)+1:end,1:size(Bmat,2)] .= Bmat
    op.jacob[1:size(Bmat,2),end-size(Bmat,1)+1:end] .= Bmat'
    # apply Dirichlet BC
    for i=eachindex(op.Dirichlet_BC)
        bcdof = Array{Int64,1}(undef, 0)
        bcdof = vcat(bcdof, findall(op.mesh.controlPoints[1,:].==op.Dirichlet_BC[i].x_val))
        for j ∈ op.Dirichlet_BC[i].comp
            bcdof .+= (j-1)*op.mesh.numBasis
            bcval = Array{Float64,1}(undef, 0)
            bcval = vcat(bcval, op.Dirichlet_BC[i].op_val)
            op.resid .-= op.stiff[:,bcdof]*bcval
            op.resid[bcdof] .= bcval
            op.stiff[bcdof, :] .= 0.0 
            op.stiff[:, bcdof] .= 0.0 
            op.jacob[bcdof,:] .= 0.0
            op.jacob[:,bcdof] .= 0.0
            op.stiff[bcdof, bcdof] = sparse(I, length(bcdof), length(bcdof))
            op.jacob[bcdof, bcdof] = sparse(I, length(bcdof), length(bcdof))
        end
    end
    return nothing
end

"""
Integrate the (consistant) global mass matrix over the mesh, given a density.
"""
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
