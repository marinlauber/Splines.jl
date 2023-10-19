using StaticArrays

abstract type AbstractODEOperator end

struct ODEOperator <: AbstractODEOperator
    x :: AbstractArray{T}
    resid :: AbstractArray{T}
    jacob :: AbstractArray{T}
    p :: Dict{String,T}
end

struct ODESolver <: AbstractFEOperator end


struct GeneralizedAlpha <: ODESolver
    op :: ODEOperator
    u :: 
    αm :: T
    αf :: T
    β :: T
    γ :: T
    op_cache
end


function residuals!(resid, op::GeneralizedAlpha, x::AbstractVector)
end

function solve_step!(x, solver::GeneralizedAlpha)