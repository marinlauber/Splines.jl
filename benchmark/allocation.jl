using BenchmarkTools
using StaticArrays

struct my_struc{T}
    A :: AbstractArray{T}
    function my_struc(a::AbstractArray{T}) where T
        new{T}(copy(a))
    end
end

struct my_better_struc{T,N}
    A :: SMatrix{N,N,T}
    function my_better_struc(a::SMatrix{N,N,T}) where {T,N}
        new{T,N}(copy(a))
    end
end

function my_prod(s::my_struc{T},b::AbstractArray{T}) where T
    s.A * b
end

function my_prod(s::my_better_struc{T,N},b::AbstractArray{T}) where {T,N}
    # how can I propagate the N here so that it is type stable?
    s.A * b
end

function my_prod(A::AbstractArray{T},b::AbstractArray{T}) where T
    A * b
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
