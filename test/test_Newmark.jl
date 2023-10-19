using Plots

T = 1
Δt = 0.1
time = collect(0.0:Δt:T);
d = cos.(2π*time/T)

a_num=[0.0]
v_num = [0.0]

γ = 0.5
β = γ/2

dⁿ = d[1]; dⁿ⁺¹ = d[2]; vα = 0.0; aα = -2π
for k in 2:length(time)
    global dⁿ, dⁿ⁺¹, aⁿ, vⁿ, aα, vα
    aⁿ = aα; vⁿ = vα
    #position
    dⁿ = d[k-1]; dⁿ⁺¹ = d[k]

    # update velocity and acceleration
    aα = 1.0/(β*Δt^2)*(dⁿ⁺¹- dⁿ- Δt*vⁿ) - (1-2*β)/(2*β)*aⁿ
    vα = γ/(β*Δt)*(dⁿ⁺¹-dⁿ) + (1-γ/β)*vⁿ + Δt*(1-γ/(2*β))*aⁿ
    push!(a_num, aα)
    push!(v_num, vα)
    
end
Plots.plot(time./T, d, label="position")
Plots.plot!(time./T, v_num, label="u̇ numerical")
Plots.plot!(time./T, -2π.*sin.(2π*time/T), label="u̇ analytical")
Plots.plot!(time./T, a_num, label="ü numerical")
Plots.plot!(time./T, -4π.*cos.(2π*time/T), label="ü analytical")
