# Boundary module
struct Boundary1D
    type::String #can be "Dirichlet, Neumann or Robin"
    x_val::Float64 #boundary point in the physical space
    u_val::Float64 #boundary point in the parameter space
    op_val::Float64 #boundary operator value
    comp::Int64 #component of the variable where the BC is applied
    function Boundary1D(type::String, x_val::Float64, u_val::Float64, op_val::Float64; comp=1)
        new(type, x_val, u_val,  op_val, comp)
    end
end
function Boundary1D(type::String, x_val::Float64, op_val::Float64; comp=1)
    return Boundary1D(type, x_val, x_val, op_val; comp=comp)
end