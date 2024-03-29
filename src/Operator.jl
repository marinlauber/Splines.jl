# Abstract FEA operator and time integrator
abstract type AbstractFEOperator end
abstract type GeneralizedAlpha end
abstract type Newmark end
"""
    StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC=[], Neumann_BC=[];
                     ρ::Function=(x)->-1.0, f=Array, T=Float64)

Defines a finite element operator for a beam element, with the following fields:
    x     : Vector of the current solution
    resid : Vector of residuals
    ext   : Vector of external force
    stiff : Stiffness matrix
    jacob : Jacobian matrix
    mass  : Mass or mass matrix, depending on the problem
    mesh  : Isogeometric mesh
    gauss_rule    : Gauss quadrature rule
    Dirichlet_BC  : Vector of Dirichlet boundary conditions
    Neumann_BC    : Vector of Neumann boundary conditions
    EI : Coefficient of bending stiffness
    EA : Coefficient of axial stiffness
"""
struct StaticFEOperator{T,Vf<:AbstractArray{T},Mf<:AbstractArray{T}} <: AbstractFEOperator
    x     :: Vf
    resid :: Vf
    ext   :: Vf
    stiff :: Mf
    jacob :: Mf
    mass  :: Union{T,Mf}
    mesh  :: Mesh
    gauss_rule   :: GaussQuad
    Dirichlet_BC :: Vector{Boundary1D}
    Neumann_BC   :: Vector{Boundary1D}
    EI :: Function
    EA :: Function
    ρ :: Union{Function,Nothing} # density as a function of curvilnear coordinate
    g :: Union{Function,Nothing} # gravitational acceleration
    function StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC=[], Neumann_BC=[];
                              ρ=nothing, g=nothing,
                              f=Array, T=Float64)
        # generate
        Nd = 2*mesh.numBasis+length(Neumann_BC); Ng = (Nd,Nd)
        x, r, q = zeros(T,Nd) |> f, zeros(T,Nd) |> f, zeros(T,Nd) |> f
        K, J, M = zeros(T,Ng) |> f, zeros(T,Ng) |> f, zeros(T,Ng) |> f
        # no mass if not provided
        global_mass!(M, mesh, ρ, gauss_rule)
        new{T,typeof(x),typeof(K)}(x,r,q,K,J,M,mesh,gauss_rule,Dirichlet_BC,Neumann_BC,F2F(EI),F2F(EA),ρ,g)
    end
end
F2F(a::Function) = a
F2F(a::Number) = (x)->a
"""
    DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC=[], Neumann_BC=[]; 
                      ρ::Function=(x)->-1.0, f=Array, ρ∞=1.0, T=Float64, I=GeneralizedAlpha)

Defines a finite element operator for a beam element, with the following fields:
    x     : Vector of the current solution
    resid : Vector of residuals
    ext   : Vector of external force
    stiff : Stiffness matrix
    jacob : Jacobian matrix
    mass  : Mass or mass matrix, depending on the problem
    mesh  : Isogeometric mesh
    gauss_rule    : Gauss quadrature rule
    Dirichlet_BC  : Vector of Dirichlet boundary conditions
    Neumann_BC    : Vector of Neumann boundary conditions
    EI : Coefficient of bending stiffness
    EA : Coefficient of axial stiffness
    f : Type of storage for the structure
    ρ∞: Spectral radius of the time integrator
    I : Time integrator, GeneralizedAlpha or Newmark
"""
struct DynamicFEOperator{I,T,Vf<:AbstractArray{T},Mf<:AbstractArray{T}} <: AbstractFEOperator
    x    :: Vf
    resid:: Vf
    ext  :: Vf
    stiff:: Mf
    jacob:: Mf
    mass :: Mf
    mesh :: Mesh
    gauss_rule :: GaussQuad
    Dirichlet_BC :: Vector{Boundary1D}
    Neumann_BC   :: Vector{Boundary1D}
    EI :: Function
    EA :: Function
    u ::Union{AbstractVector,Tuple{Vararg{AbstractVector}}}
    αm :: T
    αf :: T
    β :: T
    γ :: T
    Δt :: Vector{T}
    ρ :: Union{Function,Nothing} # density as a function of curvilnear coordinate
    g :: Union{Function,Nothing} # gravitational acceleration
    function DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC=[], Neumann_BC=[]; 
                               ρ::Function=(x)->-1.0,  ρ∞=1.0, g=nothing, 
                               f=Array, T=Float64, I=GeneralizedAlpha)
        # time integration parameters
        αm = (2.0 - ρ∞)/(ρ∞ + 1.0);
        αf = 1.0/(1.0 + ρ∞)
        γ = 0.5 - αf + αm;
        β = 0.25*(1.0 - αf + αm)^2;
        # generate
        Nd = 2*mesh.numBasis+length(Neumann_BC); Ng = (Nd,Nd)
        x, r, q = zeros(T,Nd) |> f, zeros(T,Nd) |> f, zeros(T,Nd) |> f
        K, J, M = zeros(T,Ng) |> f, zeros(T,Ng) |> f, zeros(T,Ng) |> f
        # can be made nicer
        global_mass!(M, mesh, ρ, gauss_rule) #never changes between time steps
        new{I,T,typeof(x),typeof(K)}(x,r,q,K,J,M,mesh,gauss_rule,Dirichlet_BC,Neumann_BC,F2F(EI),F2F(EA),
                                      (zero(r),zero(r),zero(r)),αm,αf,β,γ,[0.0],ρ,g)
    end
end
"""
    residual(resid, x, force, op::AbstractFEOperator)

Compute the residuals of a gicen operator `op`, using an initial solution `x` and an 
external force `force`.
"""
function residual!(resid, x, force, op::AbstractFEOperator)
    # update the jacobian, the residual and the external force
    integrate!(op, x, force)

    # apply BC
    applyBC!(op)

    # assign residual
    resid .= op.stiff*x - op.ext
    return nothing
end


"""
    jacobian!(jacob, x, force, op::AbstractFEOperator)

Compute the Jacobian of a  given operator `op`, using an initial solution `x` and an 
external force `force`.
"""
function jacobian!(jacob, x, force, op::AbstractFEOperator)
    # update the jacobian, the residual and the external force
    integrate!(op, x, force)

    # apply BC
    applyBC!(op)

    # assign jacobian
    jacob .= op.jacob
    return nothing
end

"""
    integrate!(op::AbstractFEOperator, x0::Vector{T}, fx)

Integrate the operator `op` given an initial solution `x0` and an external force. 
This performs the element-wise integration of the different term in the governing equations.
"""
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  29.019 μs …  2.566 ms  ┊ GC (min … max):  0.00% … 97.31%
#  Time  (median):     30.990 μs              ┊ GC (median):     0.00%
#  Time  (mean ± σ):   37.714 μs ± 87.849 μs  ┊ GC (mean ± σ):  12.85% ±  5.44%

#   ▃▆█▇▅▃▂▃▃▃▂▁                                                ▂
#   █████████████▇▇▆▅▆▇▆▅▅▆▅▅▄▄▄▆▆▆▆▆▅▄▅▆▇█▇▆▆▆▆▅▄▅▇█▇▆▅▆▁▅▆██▇ █
#   29 μs        Histogram: log(frequency) by time      68.9 μs <

#  Memory estimate: 150.39 KiB, allocs estimate: 1160.
function integrate!(op::AbstractFEOperator, x0::Vector, fx)
    # reset
    op.stiff .= 0.; op.jacob .= 0.; op.ext .= 0.;
    off = op.mesh.numBasis
    # precompute bernstein basis
    B, dB, ddB = bernsteinBasis(op.gauss_rule.nodes, op.mesh.degP[1])
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

        # integrate on element
        for iGauss = 1:length(op.gauss_rule.nodes)
            #compute the rational basis
            RR = N_mat[iGauss,:].* op.mesh.weights[curNodes]
            dR = dN_mat[iGauss,:].* op.mesh.weights[curNodes]
            ddR = ddN_mat[iGauss,:].* op.mesh.weights[curNodes]
            w_sum = sum(RR)
            dw_xi = sum(dR)
            dR = dR/w_sum - RR*dw_xi/w_sum^2
            ddR = ddR/w_sum - 2*dR*dw_xi/w_sum^2 - RR*sum(ddR)/w_sum^2 + 2*RR*dw_xi^2/w_sum^3

            #compute the derivatives w.r.t to the physical space
            dxdxi = dR' * op.mesh.controlPoints[1, curNodes]
            Jac_par_phys = det(dxdxi)
            RR /= w_sum
            phys_pt = RR' * op.mesh.controlPoints[1, curNodes]

            # compute linearised terms using the current solution
            du0dx = dR' * x0[curNodes]
            dw0dx = dR' * x0[curNodes.+off]

            # external force at physical point fx{size{2,numElem*numGauss}]}
            fi = fx[:,(iElem-1)*length(op.gauss_rule.nodes)+iGauss]; gravity!(op, fi, phys_pt, op.g)
            op.ext[curNodes     ] += Jac_par_phys * Jac_ref_par * fi[1] * RR * op.gauss_rule.weights[iGauss]
            op.ext[curNodes.+off] += Jac_par_phys * Jac_ref_par * fi[2] * RR * op.gauss_rule.weights[iGauss]

            # compute the different terms
            op.stiff[curNodes, curNodes] += Jac_ref_par * Jac_par_phys * op.EA(phys_pt) * (dR*dR') * op.gauss_rule.weights[iGauss]
            op.stiff[curNodes, curNodes.+off] += 0.5 * Jac_ref_par * Jac_par_phys * op.EA(phys_pt) * (dw0dx) * (dR*dR') * op.gauss_rule.weights[iGauss]
            op.stiff[curNodes.+off, curNodes.+off] += Jac_ref_par * Jac_par_phys^2 * op.EI(phys_pt) * (ddR*ddR') * op.gauss_rule.weights[iGauss]
            op.stiff[curNodes.+off, curNodes.+off] += 0.5 * Jac_ref_par * Jac_par_phys * op.EA(phys_pt) * (du0dx + dw0dx^2) * (dR*dR') * op.gauss_rule.weights[iGauss]
            
            # only different entry of Jacobian
            op.jacob[curNodes.+off, curNodes.+off] += Jac_ref_par * Jac_par_phys * op.EA(phys_pt) * (dw0dx^2) * (dR*dR') * op.gauss_rule.weights[iGauss]
        end
    end
    # enforce symmetry K21 -> K12
    op.stiff[off+1:2off,1:off] .= op.stiff[1:off,off+1:2off]
    # form jacobian
    op.jacob .+= op.stiff
end

"""
    gravity!

Add gravity contribution to the external force vector
"""
gravity!(op::AbstractFEOperator,f_ext,ξ,::Nothing) = nothing
gravity!(op::AbstractFEOperator,f_ext,ξ,::Function) = for i ∈ 1:2
    f_ext[i] += op.g(i,ξ)*op.ρ(ξ)
end

using SparseArrays: sparse
function applyBC!(op::AbstractFEOperator)
    # point to the part we modify
    Bmat = @view op.stiff[end-length(op.Neumann_BC)+1:end,1:size(op.stiff,2)]
    # apply Neumann BC
    for i=eachindex(op.Neumann_BC)
        off = (op.Neumann_BC[i].comp-1)*op.mesh.numBasis
        cdof_neu = findall(op.mesh.controlPoints[1,:].==op.Neumann_BC[i].x_val)
        for iElem=1:op.mesh.numElem
            curNodes = op.mesh.elemNode[iElem]
            if cdof_neu[1] in curNodes
                # nodes = gauss_rule.nodes
                nodes = [2*op.Neumann_BC[i].u_val-1] # map into parameter space
                # compute the Bernstein basis functions and derivatives
                B, dB, _ = bernsteinBasis(nodes, op.mesh.degP[1])
                uMin = op.mesh.elemVertex[iElem, 1]
                uMax = op.mesh.elemVertex[iElem, 2]
                Jac_ref_par = (uMax-uMin)/2

                #compute the (B-)spline basis functions and derivatives with Bezier extraction
                N_mat = B * op.mesh.C[iElem]'
                dN_mat = dB * op.mesh.C[iElem]'/Jac_ref_par

                #compute the rational spline basis
                numNodes = length(curNodes)
                localLagrange = zeros(numNodes)
                for iGauss = 1:length(nodes)
                    #compute the rational basis
                    RR = N_mat[iGauss,:].* op.mesh.weights[curNodes]
                    dR = dN_mat[iGauss,:].* op.mesh.weights[curNodes]
                    w_sum = sum(RR)
                    dw_xi = sum(dR)
                    dR = dR/w_sum - RR*dw_xi/w_sum^2

                    #compute the Jacobian of the transformation from parameter to physical space
                    dxdxi = dR' * op.mesh.controlPoints[1, curNodes]
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
        bcdof = Array{Int16,1}(undef, 0)
        bcdof = vcat(bcdof, findall(op.mesh.controlPoints[1,:].==op.Dirichlet_BC[i].x_val))
        for j ∈ op.Dirichlet_BC[i].comp
            bcdof .+= (j-1)*op.mesh.numBasis
            bcval = Array{eltype(op.stiff),1}(undef, 0)
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
global_mass!(mass::AbstractArray{T}, mesh::Mesh{T}, ::Nothing, gauss_rule) where T = (mass = zero(T))
function global_mass!(mass::AbstractArray{T}, mesh::Mesh{T}, ρ::Function, gauss_rule) where {T}
    B, dB = bernsteinBasis(gauss_rule.nodes, mesh.degP[1])
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
        localMass = zeros(T, numNodes, numNodes)
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
        end
        mass[curNodes, curNodes] += localMass
        mass[curNodes.+mesh.numBasis, curNodes.+mesh.numBasis] += localMass
    end
    return nothing
end
