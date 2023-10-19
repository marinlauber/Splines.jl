using Splines
using LinearAlgebra
using SparseArrays
include("../src/TimeIntegration.jl")

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0
A = 0.1
I = 1e-3
E = 1000.0
L = 1.0
EI = E*I #1.0
EA = E*A #10.0
f(s) = [0.0,0.0] # s is curvilinear coordinate
density(ξ) = A*1.
P = EI/20
exact_sol(x) = P.*x.^2/(6EI).*(3 .- x) # fixed - free (Ponts Load)

# natural frequencies
ωₙ = [1.875, 4.694, 7.855]
fhz = ωₙ.^2.0.*√(EI/(density(0.0)*L^4))/(2π)

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = [
    Boundary1D("Neumann", ptLeft, 0.0; comp=1),
    Boundary1D("Neumann", ptLeft, 0.0; comp=2)
]

# make a problem
p = EulerBeam(EI, EA, f, mesh, gauss_rule, Dirichlet_BC, Neumann_BC)

## Time integration
ρ∞ = 0.5; # spectral radius of the amplification matrix at infinitely large time step
αf = ρ∞/(ρ∞ + 1.0)
αm = (2*ρ∞ - 1.0)/(ρ∞ + 1.0)
γ = 0.5 - αm + αf
β = 0.25*((1.0 - αm + αf)^2)
# unconditional stability αm ≥ αf ≥ 1/2

# time steps
Δt = 0.01
T = 20.0/fhz[1]
time = collect(0.0:Δt:T);
Nₜ = length(time);

# unpack variables
@unpack x, resid, jacob = p
M = spzero(jacob)
stiff = zeros(size(jacob))
fext = zeros(size(resid))
M = global_mass!(M, mesh, density, gauss_rule)

# initialise
a0 = zeros(size(resid))
dⁿ = u₀ = zero(a0);
vⁿ = zero(a0);
aⁿ = zero(a0);
u1 = zero(a0);
v1 = zero(a0);
a1 = zero(a0);

# get the results
xs = LinRange(ptLeft, ptRight, numElem+1)

# time loop
@gif for k = 2:Nₜ

    global dⁿ, vⁿ, aⁿ, F;
    global vⁿ⁺¹, aⁿ⁺¹, dⁿ⁺¹, dⁿ⁺ᵅ, vⁿ⁺ᵅ, aⁿ⁺ᵅ,u1,v1,a1;

    tⁿ   = time[k-1]; # previous time instant
    tⁿ⁺ᵅ = tⁿ + (1-αf)*Δt; # time instant for the acceleration

    # predictor (initial guess) for the Newton-Raphson scheme
    # d_{n+1}
    dⁿ⁺¹ = dⁿ;
    r₂ = 1.0; iter = 1;

    # Newton-Raphson iterations loop
    while r₂ > 1.0e-4 && iter < 100

        # compute d_{n+af}, v_{n+af}, a_{n+am}, ...
        dⁿ⁺ᵅ = dⁿ⁺¹#(αf-1.0)*dⁿ⁺¹ + αf*dⁿ;
        vⁿ⁺ᵅ = γ/(β*Δt) * (dⁿ⁺¹ - dⁿ) +
              (αf-1.0) * (γ/β-1.0) * vⁿ + 
              (αf-1.0) * (γ/2β-1.0) * Δt * aⁿ +
               αf * vⁿ;
        aⁿ⁺ᵅ = (1.0-αm) / (1.0-αf) / (β*Δt^2) * (dⁿ⁺¹ -dⁿ) + 
                (αm-1.0) / (β*Δt) * vⁿ +
                (αm-1.0) * (1.0/2β-1.0) * aⁿ +
                αm * aⁿ;

        # update stiffness and jacobian, linearised here
        Splines.update_global!(stiff, jacob, dⁿ⁺ᵅ, p.mesh, p.gauss_rule, p)
    
        # update rhs vector
        Splines.update_external!(fext, p.mesh, p.f, p.gauss_rule)
        fext[2mesh.numBasis] += P*sin(2π*fhz[1]*tⁿ⁺ᵅ);

        # # apply BCs
        jacob .= (1.0-αm)/(1.0-αf)/(β*Δt^2)*M + jacob
        resid .= stiff*dⁿ⁺ᵅ + M*aⁿ⁺ᵅ - fext
        Splines.applyBCGlobal!(stiff, jacob, resid, p.mesh, 
                               p.Dirichlet_BC, p.Neumann_BC,
                               p.gauss_rule)

        # check convergence
        r₂ = norm(resid);
        if r₂ < 1.0e-6 && break; end
        println(" rNorm : $iter ...  $r₂ ... $(sin(tⁿ⁺ᵅ))")

        # newton solve for the displacement increment
        dⁿ⁺¹ -= jacob\resid; iter += 1
    end

    # println(" rNorm : $iter ...  $r₂")
    
    # copy variables ()_{n} <-- ()_{n+1}
    @. u1 = 1.0 / (1.0 - αf) * dⁿ⁺¹ -
            αf / (1.0 - αf) * dⁿ
    @. v1 = γ / (β*Δt) * (dⁿ⁺¹ - dⁿ) -
            (γ - β) / β * vⁿ -
            (γ - 2.0 * β) / (2.0 * β) * Δt * aⁿ
    @. a1 = 1.0 / (β * Δt * Δt) * (dⁿ⁺¹ - dⁿ) -
            1.0 / (β * Δt) * vⁿ -
            (1.0 - 2.0 * β) / (2.0 * β) * aⁿ
    dⁿ = u1;
    vⁿ = v1;
    aⁿ = a1;

    # get the results
    u0 = dⁿ[1:mesh.numBasis]
    w0 = dⁿ[mesh.numBasis+1:2mesh.numBasis]
    u = xs .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    ti =round(tⁿ,digits=3)
    Plots.plot(u, w, legend=:none, xlim=(-0.5,1.5), aspect_ratio=:equal, ylims=(-1,1), 
                title="t = $ti")
    println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=4))
end
