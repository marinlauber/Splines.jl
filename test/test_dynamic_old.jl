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
IGAmesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_left = Boundary1D("Dirichlet", ptLeft, ptLeft, 0.0)
Neumann_left = Boundary1D("Neumann", ptLeft, ptLeft, 0.0)

# make a problem
p = Problem1D(EI, EA, f, t, IGAmesh, gauss_rule,
              [Dirichlet_left], [Neumann_left])

# unpack pre-allocated storage and the convergence flag
@unpack x, resid, jacob, EI, EA, f, t, mesh, gauss_rule, Dirichlet_BC, Neumann_BC = p

# warp the residual and the jacobian requires mass
# dynamic_residuals!(resid, x, spcopy(jacob), zero(x), f, t, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
# resid[2*IGAmesh.numBasis] -= P # need a custom solver as the resdiuals are different

# # compute stiffness part of the jacobian
# dynamic_jacobian!(jacob, x, f, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
# mass = spzero(jacob); a0 = zeros(2*IGAmesh.numBasis+1); # pre-allocate
# dynamic_mass!(mass, mesh, 0., Dirichlet_BC, Neumann_BC, gauss_rule)

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
a0 = zeros(2*IGAmesh.numBasis+1)
dⁿ = u₀ = zero(a0);
vⁿ = v₀ = zero(a0);
aⁿ = a₀ = zero(a0);

# storage
uNum=[]; push!(uNum, u₀);
vNum=[]; push!(vNum, v₀);
aNum=[]; push!(aNum, a₀);

# tangent stiffness and zero damping
# K = copy(jacob); M = copy(jacob);
# C = zero(K); 
# # K̂ = α₁/(β*Δt^2)*M + α₂*γ/(β*Δt)*C + α₂*K; # effective stiffness matrix
static_jacobian!(jacob, x, f, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
# K = copy(jacob)
# M = spzero(jacob)
# dynamic_jacobian!(jacob, p.x, f, 0.0, α₁, α₂, β, Δt, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
# K̂ = spcopy(jacob)

# # solve the problems
dynamic_residuals!(resid, dⁿ, spzero(jacob), aⁿ, f, t, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
resid[2*IGAmesh.numBasis] = P

# # # externaml force
# F[2IGAmesh.numBasis] = P;
# r⁰ = F - M*aⁿ - K*u₀;

# # newton solve for the displacement increment
# u₀ += K̂\r⁰
u₀ += jacob \ resid
display(u₀')
# display((K \ r⁰)')

# get the results
u0 = u₀[1:IGAmesh.numBasis]
w0 = u₀[IGAmesh.numBasis+1:2IGAmesh.numBasis]
xs = LinRange(ptLeft, ptRight, numElem+1)
u = xs .+ getSol(IGAmesh, u0, 1)
w = getSol(IGAmesh, w0, 1)
we = exact_sol(xs)
println("Error: ", norm(w .- we))
Plots.plot(u, w, label="Sol")
Plots.plot!(xs, we, label="Exact")

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

        # vertical external force at current time
        F[2IGAmesh.numBasis] = P*sin(tⁿ⁺ᵅ);

        # solve the problems
        dynamic_residuals!(resid, dⁿ⁺ᵅ, M, aⁿ⁺ᵅ, f, t, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
        # static_jacobian!(jacob, p.x, f, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
        # resid = jacob*dⁿ⁺ᵅ
        rⁿ⁺ᵅ = F - resid

        # residuals
        # rⁿ⁺ᵅ = F - M*aⁿ⁺ᵅ - K*dⁿ⁺ᵅ;
        # display(norm(resid-K*dⁿ⁺ᵅ))
        r₂ = norm(rⁿ⁺ᵅ);
        println(" rNorm : $iter ...  $r₂ ... $(sin(tⁿ⁺ᵅ))");

        # For non-linear problems the following step should be done inside the loops
        # K̂ = α₁/(β*Δt^2)*M +  α₂*K; # effective stiffness matrix
        dynamic_jacobian!(jacob, p.x, f, M, α₁, α₂, β, Δt, mesh, EI, EA, Dirichlet_BC, Neumann_BC, gauss_rule)
        K̂ = spcopy(jacob)

        # newton solve for the displacement increment
        # !!!!!!!!!!!!!!!this is useless if we have converged!!!!!!!!!!!!!!!!
        dⁿ⁺¹ += K̂\rⁿ⁺ᵅ; iter += 1
    end
    
    # copy variables ()_{n} <-- ()_{n+1}
    dⁿ = dⁿ⁺¹;
    vⁿ = vⁿ⁺¹;
    aⁿ = aⁿ⁺¹;

    # get the results
    u0 = dⁿ[1:IGAmesh.numBasis]
    w0 = dⁿ[IGAmesh.numBasis+1:2IGAmesh.numBasis]
    u = xs .+ getSol(IGAmesh, u0, 1)
    w = getSol(IGAmesh, w0, 1)
    # we = exact_sol(xs)
    # println("Error: ", norm(w .- we))
    Plots.plot(u, w, legend=:none, ylims=(-1,1))

    # store the solution
    push!(uNum, dⁿ);
    push!(vNum, vⁿ);
    push!(aNum, aⁿ);
end

# # # validation
# # H(x) = ifelse(x <= 0.0, 0.0, 1.0)
# # E = 210e9
# # ρ = 7900
# # P₀ = 10
# # Ps(t) = P₀*H(t - 1.36)
# # h=1; b=1; L=10;
# # T = 2π/(1.875^2)*√(12ρ*L^4/(EI*h^2))
# # uz(t) = (1.0 - cos(2π/T*t))*(4P₀*L^3)/(EI*b*h^3)
# # time = collect(0:0.01:10)
# # plot(time, uz.(time), label="Exact")