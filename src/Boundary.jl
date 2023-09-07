# Boundary module
struct Boundary1D
    type::String #can be "Dirichlet, Neumann or Robin"
    x_val::Float64 #boundary point in the physical space
    u_val::Float64 #boundary point in the parameter space
    α::Number #parameter value for Robin boundary condition (αu+u')
    op_val::Number #boundary operator value
    function Boundary1D(type::String, x_val::Float64, u_val::Float64, α::Number, op_val::Number)
        new(type, x_val, u_val, α, op_val)
    end
end
function Boundary1D(type::String, x_val::Float64, u_val::Float64, op_val::Number)
    return Boundary1D(type, x_val, u_val, 0., op_val)
end