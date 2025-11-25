
### ============================================================================
### FUNCTION TYPE
### ============================================================================

using LinearAlgebra

"""
λ‖⋅‖_1 
"""
struct OneNormFunction <:ClCnvxFxn
    lambda::Float64

    function OneNormFunction(lambda::Number)
        if lambda < 0 
            error("lambda constant for OneNormFunction can be negative. ")
        end
        new(convert(Float64, lambda))
    end
end

function (this::OneNormFunction)(x::AbstractVecOrMat)
    return (this.lambda)*norm(x, 1)
end

# Traits assignment and implementations for this type

function prox_trait_assign(::OneNormFunction)
    return Proxable()
end

function dval_trait_assigner(::OneNormFunction)
    return HasFenchelDual()
end


"""
Proximal operator of the one-norm is the soft thresholding operator. 
It computes: 
prox[x ↦ λ‖ηx‖_1, ρ](y)
"""
function prox!(
    ::Proxable, 
    this::OneNormFunction, 
    y::Union{AbstractVecOrMat, Number}, 
    y_out::Union{AbstractVecOrMat, Number},
    rho::Number, # prox regularization parameter
    eta::Number #  x/eta
)::Nothing
    λ = rho*eta*(this.lambda)
    for indx in eachindex(y)
        y_out[indx] = abs(y[indx])
        y_out[indx] -= λ
        y_out[indx] = max(0, y_out[indx])*sign(y_out[indx])
    end
    return nothing
end


"""
Evaluate the dual of z ↦ λ‖z‖_1, which is the indicator of set λ{x: ‖x‖_∞ ≤ 1}.
The indicator function of rescaled norm ball of the dual norm. 
"""
function dval(
    ::HasFenchelDual, 
    this::OneNormFunction, 
    x::FiniteEuclideanSpace
)::Number
    n = norm(x, Inf)
    if n > this.lambda
        return Inf64
    end
    return 0.0
end