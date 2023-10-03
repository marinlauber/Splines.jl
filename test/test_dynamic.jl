using Splines
using LinearAlgebra
using SparseArrays
include("../src/TimeIntegration.jl")

# Material properties and mesh
numElem=3
degP=3
ptLeft = 0.0
ptRight = 1.0
EI = 1.0
EA = 1.0
f(x) = 0.0
t(x) = 0.0
P = 1.0
exact_sol(x) = P.*x.^2/(6EI).*(3 .- x) # fixed - free (Ponts Load)

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = [Boundary1D("Neumann", ptLeft, 0.0; comp=2)]

# make a problem
p = Problem1D(EI, EA, f, t, mesh, gauss_rule, Dirichlet_BC, Neumann_BC)

# unpack pre-allocated storage and the convergence flag
@unpack x, resid, jacob = p

# ##
## Time integration
##
ρ = 1.0;
α₁ = (2.0 - ρ)/(ρ + 1.0);
α₂ = 1.0/(1.0 + ρ)
γ = 0.5 + α₁ - α₂;
β = 0.25*(1.0 + α₁ - α₂)^2;

# time steps
Δt = 0.1
T = 2π
time = collect(0.0:Δt:T);
Nₜ = length(time);

# initialise
a0 = zeros(2*mesh.numBasis+1)
dⁿ = u₀ = zero(a0);
vⁿ = zero(a0);
aⁿ = zero(a0);

# unpack variables
@unpack x, resid, jacob = p
M = spzero(jacob)
# tangent stiffness and residual
static_jacobian!(jacob, x, p)
# dynamic_residuals!(resid, dⁿ, spzero(jacob), aⁿ, f, t, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
dynamic_residuals!(resid, dⁿ, aⁿ, M, p)

# add external force
resid[2*mesh.numBasis] = P

# # newton solve for the displacement increment
u₀ += jacob \ resid
display(u₀')

# get the results
xs = LinRange(ptLeft, ptRight, numElem+1)

# time loop
@gif for k = 2:Nₜ

    global dⁿ, vⁿ, aⁿ, F;
    global vⁿ⁺¹, aⁿ⁺¹, dⁿ⁺¹, dⁿ⁺ᵅ, vⁿ⁺ᵅ, aⁿ⁺ᵅ;

    tⁿ⁺¹ = time[k]; # current time instal
    tⁿ   = time[k-1]; # previous time instant
    tⁿ⁺ᵅ = α₂*tⁿ⁺¹ + (1.0-α₂)*tⁿ;

    # predictor (initial guess) for the Newton-Raphson scheme
    # d_{n+1}
    dⁿ⁺¹ = dⁿ; r₂ = 1.0; iter = 1;

    # Newton-Raphson iterations loop
    while r₂ > 1.0e-6 && iter < 10
        # compute v_{n+1}, a_{n+1}, ... from "Isogeometric analysis: toward integration of CAD and FEA"
        vⁿ⁺¹ = γ/(β*Δt)*dⁿ⁺¹ - γ/(β*Δt)*dⁿ + (1.0-γ/β)*vⁿ + Δt*(1.0-γ/2β)*aⁿ;
        aⁿ⁺¹ = 1.0/(β*Δt^2)*dⁿ⁺¹ - 1.0/(β*Δt^2)*dⁿ - 1.0/(β*Δt)*vⁿ + (1.0-1.0/2β)*aⁿ;

        # compute d_{n+af}, v_{n+af}, a_{n+am}, ...
        dⁿ⁺ᵅ = α₂*dⁿ⁺¹ + (1.0-α₂)*dⁿ;
        vⁿ⁺ᵅ = α₂*vⁿ⁺¹ + (1.0-α₂)*vⁿ;
        aⁿ⁺ᵅ = α₁*aⁿ⁺¹ + (1.0-α₁)*aⁿ;

        # residuals
        # dynamic_residuals!(resid, dⁿ⁺ᵅ, spzero(jacob), aⁿ⁺ᵅ, f, t, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
        dynamic_residuals!(resid, dⁿ⁺ᵅ, aⁿ⁺ᵅ, M, p)
        resid[2mesh.numBasis] += P*sin(tⁿ⁺ᵅ);
        rⁿ⁺ᵅ = -resid; r₂ = norm(rⁿ⁺ᵅ);
        println(" rNorm : $iter ...  $r₂ ... $(sin(tⁿ⁺ᵅ))");

        # For non-linear problems the following step should be done inside the loops
        # K̂ = α₁/(β*Δt^2)*M +  α₂*K; # effective stiffness matrix
        # dynamic_jacobian!(jacob, p.x, f, spzero(jacob), α₁, α₂, β, Δt, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
        dynamic_jacobian!(jacob, dⁿ⁺ᵅ, α₁, α₂, β, Δt, M, p)

        # newton solve for the displacement increment
        # !!!!!!!!!!!!!!!this is useless if we have converged!!!!!!!!!!!!!!!!
        dⁿ⁺¹ += jacob\rⁿ⁺ᵅ; iter += 1
    end
    
    # copy variables ()_{n} <-- ()_{n+1}
    dⁿ = dⁿ⁺¹;
    vⁿ = vⁿ⁺¹;
    aⁿ = aⁿ⁺¹;

    # get the results
    u0 = dⁿ[1:mesh.numBasis]
    w0 = dⁿ[mesh.numBasis+1:2mesh.numBasis]
    u = xs .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    Plots.plot(u, w, legend=:none, xlim=(-0.5,1.5), aspect_ratio=:equal, ylims=(-1,1))
end
