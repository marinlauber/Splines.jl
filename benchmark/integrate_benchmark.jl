using Splines
using BenchmarkTools
using StaticArrays

struct my_struc{T}
    A :: AbstractArray{T}
    function my_struc(a::AbstractArray{T}) where T
        new{T}(copy(a))
    end
end

struct my_better_struc{T}
    A :: SMatrix{4,4,T}
    function my_better_struc(a::SMatrix{T,N,N}) where {T,N}
        new{T}(copy(a))
    end
end

function my_prod(s::my_struc{T},b::AbstractArray{T}) where T
    s.A * b
end

function my_prod(s::my_better_struc{T},b::AbstractArray{T}) where T
    s.A * b
end

function my_prod(A::AbstractArray{T},b::AbstractArray{T}) where T
    A * b
end

function test()
    my_prod(my.A,b)
end

function test(my::my_struc{T},b::AbstractArray{T}) where T
    my_prod(my.A,b)
end

N = 4
A = @SArray rand(Float64,N,N)
b = @SArray ones(Float64,N)

my = my_struc(A)

my_better = my_better_struc(A)

@btime my_prod($my,$b) # 14.607 ns (2 allocations: 96 bytes)
@btime my_prod($A,$b) # 2.514 ns (0 allocations: 0 bytes)
@btime my_prod($my.A,$b) # 13.458 ns (2 allocations: 96 bytes)
@btime my_prod($my_better,$b) # 110.688 ns (2 allocations: 96 bytes)
@btime my_prod($my_better.A,$b) # 13.222 ns (2 allocations: 96 bytes)













# dummmy mesh
mesh, gauss_rule = Mesh1D(0.0, 1.0, 10, 3)

# make a problem
operator = DynamicFEOperator(mesh, gauss_rule, 1.0, 1.0, [], [],
                             ρ=(x)->2.0; ρ∞=0.0)
forces = zeros(2,length(uv_integration(operator))); forces[2,:] .= 1.0
x0 = zero(operator.x)

# @benchmark Splines.integrate_fast!($operator, $x0, $forces)
# @benchmark integrate!($operator, $x0, $forces)
@benchmark solve_step!($operator, $forces, $0.01)
