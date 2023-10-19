using Splines
using LinearAlgebra
using Plots

# size of problem
numElem=10
degP=3

# Material properties and mesh
ptLeft = 0.0
ptRight = 1.0
EI = 1.0
EA = 1.0
f(x) = [0.0, 1.0]
exact_sol(x) = f(x)[2]/(24EI).*(x .- 2x.^3 .+ x.^4) # pinned - pinned

mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = []

ρ∞ = 1. # spectral radius of the amplification matrix at infinitely large time step
αf = ρ∞/(ρ∞ + 1.0)
αm = (2*ρ∞ - 1.0)/(ρ∞ + 1.0)
γ = 0.5 - αm + αf
β = 0.25*((1.0 - αm + αf)^2)
Δt = 1.0

# make a problem
p = EulerBeam(EI, EA, f, mesh, gauss_rule, Dirichlet_BC, Neumann_BC)

# mass modal analysis
mass = zero(p.jacob)
stiff_dummy = zero(p.jacob)
fext = zero(p.resid)
dⁿ⁺¹ = zero(p.resid)
dⁿ = zero(p.resid)
vⁿ = zero(p.resid)
aⁿ = zero(p.resid)
vαf = zero(p.resid)
aαm = zero(p.resid)

# recompute mass
Splines.global_mass!(mass, p.mesh, x->1.0, p.gauss_rule)

r₂ = 1.0; iter =0
println("Starting iterations")
while iter < 100

    global dⁿ, dⁿ⁺¹, aαm, aαf, iter, r₂


    @. vαf = (γ) / (β * Δt) * (dⁿ⁺¹ - dⁿ) +
             (αf - 1.0) * (γ - β) / β * vⁿ +
             (αf - 1.0) * (γ - 2.0*β) / (2.0 * β) * Δt * aⁿ +
              αf * vⁿ

    @. aαm = (1.0 - αm) / (1.0 - αf) / (β * Δt * Δt) * (dⁿ⁺¹ - dⁿ) +
             (αm - 1.0) / (β * Δt) * vⁿ +
             (αm - 1.0) * (1.0 - 2.0*β) / (2.0 * β) * aⁿ +
              αm * aⁿ

    # stiffness
    Splines.update_global!(stiff_dummy, p.jacob, p.x, p.mesh, p.gauss_rule, p)

    # external
    Splines.update_external!(fext, p.mesh, p.f, p.gauss_rule)


    println("dⁿ⁺¹: $(norm(dⁿ⁺¹))")
    println("vαf: $(norm(vαf))")
    println("aαm: $(norm(aαm))")

    # applpy BC
    p.resid .= ((1.0-αm)/(1.0-αf)/(β*Δt*Δt)*mass + stiff_dummy)*dⁿ⁺¹ - mass*aαm - fext
    p.jacob .= (1.0-αm)/(1.0-αf)/(β*Δt*Δt)*mass + p.jacob
    Splines.applyBCNewton!(p.jacob, p.resid, p.mesh, p.Dirichlet_BC, p.Neumann_BC, p.gauss_rule)

    r₂ = norm(p.resid)
    println("Residual: ", r₂)
    if r₂ < 1.0e-6 && break; end

    # check that this is fine
    dⁿ⁺¹ -= p.jacob \ p.resid
    iter += 1
    println("")
end
# # recompute resdiuals
# dⁿ = result
# p.resid .= ((1.0-αm)/(1.0-αf)/(β*Δt*Δt)*mass + stiff_dummy)*dⁿ + mass*aαm - fext
# p.jacob .= (1.0-αm)/(1.0-αf)/(β*Δt*Δt)*mass + p.jacob
# Splines.applyBCNewton!(p.jacob, p.resid, p.mesh, p.Dirichlet_BC, p.Neumann_BC, p.gauss_rule)

println("Iterations $iter")

# get the solution
u0 = dⁿ⁺¹[1:mesh.numBasis]
w0 = dⁿ⁺¹[mesh.numBasis+1:2mesh.numBasis]
x = LinRange(ptLeft, ptRight, numElem+1)
u = x .+ getSol(mesh, u0, length(x))
w = getSol(mesh, w0, length(x))
we = exact_sol(x)
println("Error: ", norm(w .- we))
Plots.plot(u, w, label="Sol")
Plots.plot!(x, we, label="Exact")


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