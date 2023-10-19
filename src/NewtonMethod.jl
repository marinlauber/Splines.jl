using LinearAlgebra
using StaticArrays
using Plots
using IterativeSolvers
# using MatrixMarket

L₂(x) = sqrt(sum(abs2,x))/length(x)

"""
    gmres(A,b,m)

Do `m` iterations of GMRES for the linear system `A`*x=`b`. Returns
the final solution estimate x and a vector with the history of
residual norms.
"""
function gmres(A,b,m)
    n = length(b)
    Q = zeros(n,m+1)
    Q[:,1] = b/norm(b)
    H = zeros(m+1,m)
    j=1
    # Initial solution is zero.
    x = 0
    residual = [norm(b);zeros(m)]
    while L₂(residual[j]) > 1e-16 && j <= m
        # Next step of Arnoldi iteration.
        v = A*Q[:,j]
        for i in 1:j
            H[i,j] = dot(Q[:,i],v)
            v -= H[i,j]*Q[:,i]
        end
        H[j+1,j] = norm(v)
        Q[:,j+1] = v/H[j+1,j]

        # Solve the minimum residual problem.
        r = [norm(b); zeros(j)]
        z = H[1:j+1,1:j] \ r
        x = Q[:,1:j]*z
        residual[j+1] = norm( A*x - b )
        j+=1
    end
    return x,residual
end


"""
    backsub(A,b)

Solve the upper triangular linear system with matrix `A` and
right-hand side vector `b`.
"""
function backsub(A,b)
    n = size(A,1)
    x = zeros(n)
    x[n] = b[n]/A[n,n]
    for i in n-1:-1:1
        s = sum( A[i,j]*x[j] for j in i+1:n )
        x[i] = ( b[i] - s ) / A[i,i]
    end
    return x
end

function Relaxation(xᵏ, R, k=0)
    ω = 0.05
    xᵏ⁺¹=0.
    rᵏ = R(xᵏ)
    resid=[L₂(rᵏ)]
    while L₂(rᵏ) > 1e-16 && k < 1000
        xᵏ⁺¹ = xᵏ .+ ω.*rᵏ
        k+=1; xᵏ=xᵏ⁺¹
        rᵏ = R(xᵏ)
        push!(resid,L₂(rᵏ))
    end
    return xᵏ⁺¹,resid
end

function Newton(xᵏ, R, R′; n=0)
    while L₂(f(xᵏ)) > 1e-16 && n < 100
        rᵏ = R(xᵏ)
        Δxᵏ = R′(xᵏ)\-rᵏ
        xᵏ .+= Δxᵏ
        n += 1
    end
    println(n)
    return xᵏ
end


function IQNILS(xᵏ::Array{Float64},R::Function;k=0,ω=0.5,N=size(xᵏ,1))
    rᵏ=R(xᵏ); rⁱ=copy(rᵏ) 
    xⁱ=copy(xᵏ)
    resid=[L₂(rᵏ)]
    while L₂(rᵏ) > 1e-16 && k < 2N
        if k==0
            xᵏ .= xᵏ .- ω*rᵏ
        else
            Vᵏ = rⁱ .- rᵏ
            Wᵏ = xⁱ .- xᵏ
            rⁱ = hcat(rᵏ,rⁱ)
            xⁱ = hcat(xᵏ,xⁱ)
            # here we cannot have an undetermined system
            Vᵏ = Vᵏ[:,1:min(k,N)]
            Wᵏ = Wᵏ[:,1:min(k,N)]
            # solve least-square with Housholder QR decomposition
            Qᵏ,Rᵏ = qr(Vᵏ)
            cᵏ = backsub(Rᵏ,-Qᵏ'*rᵏ)
            xᵏ.+= Wᵏ*cᵏ #.+ rᵏ #not sure
        end
        rᵏ = R(xᵏ); k+=1
        push!(resid,L₂(rᵏ))
    end
    return xᵏ,resid
end


abstract type AbstractCoupling end

# struct ConstantRelaxation <: AbstractCoupling
#     ω :: Float64
#     r :: AbstractArray{Float64}
#     x :: AbstractArray{Float64}
#     function ConstantRelaxation(N::Int64;ω::Float64=0.5)
#         new(ω,zeros(N),zeros(N))
#     end
# end
# function update(cp::ConstantRelaxation, xᵏ, rᵏ)
#     x = (1-cp.ω)*xᵏ .+ cp.ω*cp.r; cp.x .= xᵏ
#     r = (1-cp.ω)*rᵏ .+ cp.ω*cp.r; cp.r .= rᵏ
#     return x, r
# end

struct IQNCoupling <: AbstractCoupling
    ω :: Float64
    r :: AbstractArray{Float64}
    x :: AbstractArray{Float64}
    V :: AbstractArray{Float64}
    W :: AbstractArray{Float64}
    iter :: Vector{Int64}
    function IQNCoupling(N::Int64;ω::Float64=0.5)
        new(ω,zeros(N),zeros(N),zeros(N,N),zeros(N,N),[0])
    end
end
function update(cp::IQNCoupling, xᵏ, rᵏ)
    if cp.iter[1]==0
        # store variable and residual
        cp.x .= xᵏ; cp.r .= rᵏ
        # relaxation update
        xᵏ .+= cp.ω*rᵏ
        cp.iter[1] = 1
    else
        # roll the matrix to make space for new column
        roll!(cp.V); roll!(cp.W)
        cp.V[:,1] = rᵏ .- cp.r; cp.r .= rᵏ
        cp.W[:,1] = xᵏ .- cp.x; cp.x .= xᵏ
        # solve least-square problem with Housholder QR decomposition
        Qᵏ,Rᵏ = qr(@view cp.V[:,1:min(cp.iter[1],N)])
        cᵏ = backsub(Rᵏ,-Qᵏ'*rᵏ)
        xᵏ.+= (@view cp.W[:,1:min(cp.iter[1],N)])*cᵏ #.+ rᵏ #not sure
        cp.iter[1] = cp.iter[1] + 1
    end
    return xᵏ
end
roll!(A::AbstractArray) = (A[:,2:end] .= A[:,1:end-1])


function IQNILS2(x⁰::Array{Float64},R::Function;k=0,ω=0.05,N=size(x⁰,1))
    r⁰=R(x⁰); xᵏ=copy(x⁰)
    resid=[L₂(r⁰)]; rᵏ = copy(r⁰)
    xᵏ⁻¹ = zero(xᵏ); rᵏ⁻¹ = zero(rᵏ)
    Vᵏ=0; Wᵏ=0
    while L₂(rᵏ) > 1e-16 && k < 2N
        if k==0
            xᵏ .= x⁰ .+ ω*rᵏ
            Vᵏ = R(xᵏ)-rᵏ
            Wᵏ = xᵏ .- x⁰
            xᵏ⁻¹ = x⁰
        else
            Δrᵏ = rᵏ .- rᵏ⁻¹
            Δxᵏ = xᵏ .- xᵏ⁻¹
            Vᵏ = hcat(Δrᵏ,Vᵏ)
            Wᵏ = hcat(Δxᵏ,Wᵏ)
            # here we cannot have an undetermined system
            Vᵏ = Vᵏ[:,1:min(k,N)]
            Wᵏ = Wᵏ[:,1:min(k,N)]
            # solve least-square with Housholder QR decomposition
            Qᵏ,Rᵏ = qr(Vᵏ)
            cᵏ = backsub(Rᵏ,-Qᵏ'*rᵏ)
            xᵏ⁻¹ .= xᵏ
            xᵏ.+= Wᵏ*cᵏ #.+ rᵏ #not sure
        end
        rᵏ⁻¹ = rᵏ
        rᵏ = R(xᵏ); k+=1
        push!(resid,L₂(rᵏ))
    end
    return xᵏ,resid
end

# f(x) = [2x[1]+3x[2], 5x[1]+4x[2]^3]
# df(x) = [[2, 3] [5, 12x[2]^2]]

# x0 = [4.,-2]
# x = Newton(x0, f, df)
# display(x)

# x0 = [4.,-2]
# x = Relaxation(x0, f)
# println(x)

# x0 = [4.,-2.]
# sol = IQNILS(x0,f);
# display(sol)



# A = rand(float(1:9),6,6)
# b = rand(float(1:9),6)
# f(x) = A*x - b
# df(x) = A
# # sol = Newton(zeros(6), f, df)
# sol = IQNILS(zeros(6), f);
# sol



# N = 10
# ρ = collect(range(1e-4,1e-2,N))
# C = collect(range(1e-7,1e-5,N))
# k = collect(range(1e-1,1e-3,N))
# μ = 1e-3
# du = k[2:end]
# d = ones(N); d[2:end-1]=-k[3:end].+k[1:end-2]
# dl = k[1:end-1]
# A = diagm(ρ.*C) - μ/2.0.*Tridiagonal(dl,d,du)
# A[1,:]=A[end,:]=A[:,1]=A[:,end].=0.0
# A[1,1]=A[end,end]=1.0
# b = ρ.*C
# f(x) = A*x - b
# df(x) = A

# sol  = Newton(ones(N), f, df)

# N = 10
# A = Tridiagonal(ones(N-1),-2*ones(N),ones(N-1))
# A[[1,end],:].=0.;
# A[:,[1,end]].=0;
# A[1,1]=A[end,end]=1; A=A*N
# b = 2*ones(N); b[[1,end]] .= 4
# f(x) = A*x .- b
# df(x) = A

# # sol = Newton(ones(N), f, df)
# sol,res = IQNILS(zeros(N), f)
# plot(sol)

# non-symmetric matrix wih know eigenvalues
N = 24
λ = 10 .+ (1:N)
# A = triu(rand(N,N),1) + diagm(λ)
A = rand(N,N) + diagm(λ)
b = rand(N);

# IQNILS method
f(x) = b - A*x

@time sol,r1 = gmres(A,b,100);
@assert L₂(f(sol)) < 1e-6

x0 = copy(b)
sol,history = IterativeSolvers.gmres(A,b;log=true, reltol=1e-16)
r3 = history.data[:resnorm]

x0 = copy(b)
@time sol,r2 = IQNILS2(x0, f; ω=0.05)
# @assert L₂(f(sol)) < 1e-6
println("IQNILS: ", sol)

p = plot(r1, marker=:s, xaxis=:log10, yaxis=:log10, label="GMRES",
         xlabel="Iteration", ylabel="Residual",
         xlim=(1,200), ylim=(1e-16,1e2), legend=:bottomleft)
plot!(p, r3, marker=:d, xaxis=:log10, yaxis=:log10, label="IterativeSolvers.GMRES")
plot!(p, r2, marker=:o, xaxis=:log10, yaxis=:log10, label="IQN-ILS", legend=:bottomleft)

x0 = copy(b)
@time sol,resid_relax = Relaxation(x0, f, 0.05)
# @assert L₂(f(sol)) < 1e-6
plot!(p, resid_relax, marker=:o, xaxis=:log10, yaxis=:log10, label="Relaxation", legend=:bottomleft)

IQNSolver = IQNCoupling(N;ω=0.05)
xᵏ = copy(b); rᵏ = f(xᵏ); k=1; resid=[]; sol=[]
@time while L₂(rᵏ) > 1e-16 && k < 2N
    global xᵏ, rᵏ, k, resid, sol
    xᵏ = update(IQNSolver, xᵏ, rᵏ)
    rᵏ = f(xᵏ)
    push!(sol,xᵏ)
    push!(resid,L₂(rᵏ))
    k+=1
end

plot!(p, resid, marker=:o, xaxis=:log10, yaxis=:log10, label="IQN-ILS struct", legend=:bottomleft)
savefig(p, "GMRESvsIQNILS.png")
p


# # # test on 1D heat equation
# N = 10
# A = Tridiagonal(ones(N-1),-2*ones(N),ones(N-1))
# A[1,:]=A[end,:].=0
# A[:,1]=A[:,end].=0.0
# A[1,1]=A[end,end]=1.0
# b = 2*ones(N)
# P = diagm(1.0./diag(A))
# f(x) = b - P*A*x

# x0 = zeros(N)
# @time sol,r2 = IQNILS(x0, f; ω=0.05)
# println("IQNILS: ", sol)

# sol,history = IterativeSolvers.gmres(A,b;log=true,reltol=1e-16)


# # matrix market test
# M = MatrixMarket.mmread("Workspace/shl____0.mtx")
# b = rand(size(M,1))
# sol,history = IterativeSolvers.gmres(M,b;log=true,reltol=1e-16)
# residuals = history.data[:resnorm]

