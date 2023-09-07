using FastGaussQuadrature
struct GaussQuad
    nodes::Array{Float64,1}
    weights::Array{Float64,1}
end

"""
Generates the Gauss-Legendre points of order n
"""
function genGaussLegendre(n)
    nodes, weights = gausslegendre(n)
    gauss_rule = GaussQuad(nodes, weights)
    return gauss_rule
end