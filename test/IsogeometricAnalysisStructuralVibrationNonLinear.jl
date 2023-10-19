using Splines
using LinearAlgebra
using Plots

# size of problem
numElem=4
degP=3

# Material properties and mesh
ptLeft = 0.0
ptRight = 1.0
A = 0.1
I = 1e-3
E = 1000.0
L = 1.0
EI = E*I #1.0
EA = E*A #10.0
f(s) = [0.0,0.0] # s is curvilinear coordinate
density(ξ) = A*0.1
P = EI/2
exact_sol(x) = P.*x.^2/(6EI).*(3 .- x)# fixed - free (Ponts Load)

# make a mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# # boundary conditions
# Dirichlet_BC = [
#     Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
#     Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
#     Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
#     Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
# ]
# Neumann_BC = []
# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = [
    Boundary1D("Neumann", ptLeft, 0.0; comp=1),
    Boundary1D("Neumann", ptLeft, 0.0; comp=2)
]

# HHT-α coefficients
ρ∞ = 0.5 # spectral radius of the amplification matrix at infinitely large time step
αf = ρ∞/(ρ∞ + 1.0)
αm = (2*ρ∞ - 1.0)/(ρ∞ + 1.0)
γ = 0.5 - αm + αf
β = 0.25*((1.0 - αm + αf)^2)
Δt = 1.0

# make a problem
p = EulerBeam(EI, EA, f, mesh, gauss_rule, Dirichlet_BC, Neumann_BC)

# mass modal analysis
mass = zero(p.jacob)
stiff = zero(p.jacob)
fext = zero(p.resid)
dⁿ⁺¹ = zero(p.resid)
dⁿ = zero(p.resid)
vⁿ = zero(p.resid)
aⁿ = zero(p.resid)

# recompute mass
Splines.global_mass!(mass, p.mesh, density, p.gauss_rule)

# time steps
Δt = 0.10
T = 2
time = collect(0.0:Δt:T);
Nₜ = length(time);

# time loop
@gif for k = 2:Nₜ

    global dⁿ, dⁿ⁺¹, vⁿ, aⁿ, iter, r₂, vαf, aαm

    tⁿ  = time[k-1]; # previous time instant
    tαf = tⁿ + (1.0-αf)*Δt

    dⁿ⁺¹ = dⁿ; r₂ = 1.0; iter = 1;
    while iter < 1000

        # update velocity and acceleration
        vαf = @. γ/(β*Δt) * (dⁿ⁺¹-dⁿ) +
                (αf-1.0) * (γ/β-1.0) * vⁿ +
                (αf-1.0) * (γ/2β-1.0) * Δt * aⁿ +
                 αf * vⁿ

        aαm = @. (1.0-αm)/(1.0-αf)/(β*Δt^2) * (dⁿ⁺¹ - dⁿ) +
                (αm - 1.0) / (β * Δt) * vⁿ +
                (αm - 1.0) * (1.0/2β - 1.0) * aⁿ +
                 αm * aⁿ

        # stiffness, linearised here
        Splines.update_global!(stiff, p.jacob, zero(dⁿ), p.mesh, p.gauss_rule, p)
        # Splines.update_global!(stiff, p.jacob, dⁿ, p.mesh, p.gauss_rule, p)

        # external
        Splines.update_external!(fext, p.mesh, p.f, p.gauss_rule)
        # fext[2mesh.numBasis] += P
        fext[2mesh.numBasis] += P*sin(2π*tαf)

        # applpy BC
        p.resid .= ((1.0-αm)/(1.0-αf)/(β*Δt^2)*mass + stiff)*dⁿ⁺¹ - mass*aαm - fext
        p.jacob .=  (1.0-αm)/(1.0-αf)/(β*Δt^2)*mass + p.jacob
        # Splines.applyBCNewton!(p.jacob, p.resid, p.mesh, p.Dirichlet_BC, p.Neumann_BC, p.gauss_rule)
        Splines.applyBCGlobal!(stiff, p.jacob, p.resid, p.mesh,
                               p.Dirichlet_BC, p.Neumann_BC,
                               p.gauss_rule)
        r₂ = norm(p.resid)
        if r₂ < 1.0e-6 && break; end
        
        # if not converged, update displacement and start again
        dⁿ⁺¹ -= p.jacob \ p.resid
        iter += 1
    end
    println(" rNorm : $iter ...  $r₂")

    # update solution
    dⁿ = dⁿ⁺¹
    # vⁿ = vαf #vⁿ
    # aⁿ = aαm #aⁿ
    # dⁿ⁺¹ = 1.0 / (1.0 - αf) * dⁿ⁺¹ -
            # αf / (1.0 - αf) * dⁿ
    vⁿ = γ / (β*Δt) * (dⁿ⁺¹ - dⁿ) -
            (γ - β) / β * vⁿ -
            (γ - 2.0 * β) / (2.0 * β) * Δt * aⁿ #vαf #vⁿ
    aⁿ = 1.0 / (β * Δt^2) * (dⁿ⁺¹ - dⁿ) -
         1.0 / (β * Δt) * vⁿ -
        (1.0 - 2.0 * β) / (2.0 * β) * aⁿ #aαm #aⁿ

    # get the solution
    u0 = dⁿ⁺¹[1:mesh.numBasis]
    w0 = dⁿ⁺¹[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, length(x))
    w = getSol(mesh, w0, length(x))
    we = exact_sol(x)*sin(2π*tαf)
    # println("Error: ", norm(w .- we))
    Plots.plot(x, we, label="Exact", marker=:o)
    Plots.plot!(u, w, label="IGA", xlim=(-0.5,1.5), aspect_ratio=:equal, ylims=(-1,1), 
    title="t = $tⁿ")
end

# # eigenvalue problem ( only transverse vibrartion )
# λ,V = eigen(p.jacob[mesh.numBasis+1:end,mesh.numBasis+1:end] - 
#             mass[mesh.numBasis+1:end,mesh.numBasis+1:end])

# Plots.plot()
# for mode ∈ 1:5
#     # dⁿ = V[1:mesh.numBasis,mode]
#     w0 = V[:,mode]
#     x = LinRange(ptLeft, ptRight, numElem+1)
#     w = getSol(mesh, w0, length(x))
#     Plots.plot!(x, w, label="mode $mode")
# end
# Plots.plot!()

# # true natural modes
# ωₑ = [(π*n)^2 for n ∈ 1:10]/2π