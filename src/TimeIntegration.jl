using StaticArrays
using LinearAlgebra
using Plots

abstract type AbstractIntegrator end

struct Integrator{T, Vf<:AbstractArray{T}} <: AbstractIntegrator
    ρ :: T # ρ = 0.5
    α₁:: T # α₁ = (2.0 - ρ)/(ρ + 1.0);
    α₂:: T # α₂ = 1.0/(1.0 + ρ)
    γ :: T # γ = 0.5 + α₁ - α₂;
    β :: T # β = 0.25*(1.0 + α₁ - α₂)^2;
    dⁿ :: Vf
    vⁿ :: Vf
    aⁿ :: Vf
    resid::Function
    function Integrator(ρ::T, u₀::AbstractArray{T}, resid::Function) where {T}
        α₁ = (2.0 - ρ)/(ρ + 1.0)
        α₂ = 1.0/(1.0 + ρ)
        γ = 0.5 + α₁ - α₂
        β = 0.25*(1.0 + α₁ - α₂)^2
        new{T,typeof(u₀)}(ρ,α₁,α₂,γ,β,copy(u₀),zeros(T,length(u₀)),zeros(T,length(u₀)),resid)
    end
end

function (a::Integrator)(dⁿ⁺¹,dⁿ,vⁿ,aⁿ,tⁿ,tⁿ⁺¹,K̂)
    # time step and predicted time
    Δt = tⁿ⁺¹ - tⁿ
    tⁿ⁺ᵅ = a.α₂*tⁿ⁺¹ + (1.0-a.α₂)*tⁿ

    # predictor (initial guess) for the Newton-Raphson scheme
    r₂ = 1.0; iter = 1; vⁿ⁺¹=copy(vⁿ); aⁿ⁺¹=copy(aⁿ)
    # Newton-Raphson iterations loop
    while r₂ > 1.0e-6 && iter < 10
        # compute v_{n+1}, a_{n+1}, ... from Isogeometric analysis: toward integration of CAD and FEA
        vⁿ⁺¹ =   a.γ/(a.β*Δt)*dⁿ⁺¹ -   a.γ/(a.β*Δt)*a.dⁿ + (1.0-a.γ/a.β)*a.vⁿ + Δt*(1.0-a.γ/2a.β)*a.aⁿ
        aⁿ⁺¹ = 1.0/(a.β*Δt^2)*dⁿ⁺¹ - 1.0/(a.β*Δt^2)*a.dⁿ -  1.0/(a.β*Δt)*a.vⁿ +    (1.0-1.0/2a.β)*a.aⁿ

        # compute d_{n+af}, v_{n+af}, a_{n+am}, ...
        dⁿ⁺ᵅ = a.α₂*dⁿ⁺¹ + (1.0-a.α₂)*a.dⁿ
        vⁿ⁺ᵅ = a.α₂*vⁿ⁺¹ + (1.0-a.α₂)*a.vⁿ
        aⁿ⁺ᵅ = a.α₁*aⁿ⁺¹ + (1.0-a.α₁)*a.aⁿ

        # residuals
        rⁿ⁺ᵅ = a.resid(tⁿ⁺ᵅ ,dⁿ⁺ᵅ, vⁿ⁺ᵅ, aⁿ⁺ᵅ)
        r₂ = norm(rⁿ⁺ᵅ);

        # For non-linear problems the following step should be done inside the loops
        # K̂ = α₁/(β*Δt^2)*M + α₂*γ/(β*Δt)*C + α₂*K; # effective stiffness matrix

        # newton solve for the displacement increment
        dⁿ⁺¹ += K̂\rⁿ⁺ᵅ; iter += 1
    end
    return dⁿ⁺¹,vⁿ⁺¹,aⁿ⁺¹
end


function test()
    # mass, damping and stiffness matrices
    m₁ = 0.0; m₂ = 1.0; m₃ = 1.0;
    k₁ = 10^4; k₂ = 1.0;

    # matrix assembly
    M = [m₂ 0.0;0.0 m₃];
    C = [0.0 0.0;0.0 0.0];
    K = [k₁+k₂ -k₂; -k₂ k₂];

    # initial data
    u₀ = [0.0, 0.0];
    v₀ = [0.0, 0.0];
    a₀ = M\(-C*v₀ - K*u₀);

    # time step size
    Δt = 0.1;
    T = 2*pi;

    # constants for the time integration scheme
    # from: A time integration algorithm for structural dynamics with improved numerical dissipation: the generalized-α method
    ρ = 0.5;

    # external force
    F(t) = [k₁*sin(t), 0.0];

    # residual function
    resid(t,d,v,a) = F(t) - M*a - C*v - K*d;
    CHα = Integrator(ρ, u₀, resid)

    # time steps
    t = collect(0.0:Δt:4T);
    Nₜ = length(t);

    # storage
    uNum=[]; push!(uNum, u₀);
    vNum=[]; push!(vNum, v₀);
    aNum=[]; push!(aNum, a₀);

    # initialise
    dⁿ = u₀;
    vⁿ = v₀;
    aⁿ = a₀;

    # For non-linear problems the following step should be done inside the loops
    K̂ = CHα.α₁/(CHα.β*Δt^2)*M + CHα.α₂*CHα.γ/(CHα.β*Δt)*C + CHα.α₂*K; # effective stiffness matrix

    # time loop
    for k = 2:Nₜ

        global dⁿ, vⁿ, aⁿ

        tⁿ⁺¹ = t[k]; # current time instal
        tⁿ   = t[k-1]; # previous time instant

        # predictor (initial guess) for the Newton-Raphson scheme
        # d_{n+1}
        dⁿ⁺¹ = copy(dⁿ); r₂ = 1.0; iter = 0;
        dⁿ⁺¹,vⁿ⁺¹,aⁿ⁺¹ = CHα(dⁿ⁺¹,dⁿ,vⁿ,aⁿ,tⁿ,tⁿ⁺¹,K̂)

        # copy variables ()_{n} <-- ()_{n+1}
        dⁿ = dⁿ⁺¹;
        vⁿ = vⁿ⁺¹;
        aⁿ = aⁿ⁺¹;
        
        # store the solution
        push!(uNum, dⁿ);
        push!(vNum, vⁿ);
        push!(aNum, aⁿ);
    end
    uNum = hcat(uNum...);
    Plots.plot(t./T, uNum[1,:], color=:black)
end