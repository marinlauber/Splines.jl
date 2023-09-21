using Splines

KnotVec(cps, d) = vcat(zeros(d), range(0, size(cps, 2)-d) / (size(cps, 2)-d), ones(d))

# WaterLily constructor
degP = 4
L = 1
Points = L*[[0.,0.],[0.25,0.0],[0.5,0.0],
            [0.75,0.0],[1,0.]]
weights = [1.,1.,1.,1.,1.]
knots = KnotVec(reduce(hcat, Points), degP)

# FEA constructor
coefs = vcat(reduce(hcat, Points),vcat(zeros(size(Points,1))',weights'))
nrb = Splines.NURBS([size(coefs,2)], coefs, [knots], [degP+1])
IGAmesh = Splines.genMesh(nrb)


#
# Simply supported beam

# Material properties and mesh
EI = 1.0
EA = 1.0
f(x) = 1.0
t(x) = 0.0
exact_sol(x) = f(x)/(24EI).*(x .- 2x.^3 .+ x.^4)

# integration rule
gauss_rule = Splines.genGaussLegendre(degP+1)

# boundary conditions
Dirichlet_right = Boundary1D("Dirichlet", coefs[1,1], coefs[1,1], 0.0)
Dirichlet_left = Boundary1D("Dirichlet", coefs[end,1], coefs[end,1], 0.0)

# make a problem
p = Problem1D(EI, EA, f, t, IGAmesh, gauss_rule,
             [Dirichlet_left,Dirichlet_right])

result = static_lsolve!(p, static_residuals!, static_jacobian!)

# extract solution
u0 = result[1:IGAmesh.numBasis]
w0 = result[IGAmesh.numBasis+1:2IGAmesh.numBasis]
u = getSol(IGAmesh, u0, 100)
u += LinRange(coefs[1,1], coefs[end,1], length(u))
w = getSol(IGAmesh, w0, 100)
we = exact_sol(x)
println("Error: ", norm(w .- we))
Plots.plot(u, w, label="Sol")
Plots.plot!(x, we, label="Exact")
Plots.plot!(nrb.coefs[1,:]+u0, nrb.coefs[2,:]+w0, marker=:o, label="Control points")
