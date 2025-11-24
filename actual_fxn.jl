
### ============================================================================
### FUNCTION TYPE
### ============================================================================

using LinearAlgebra

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

function prox_trait_assign(::OneNormFunction)
    return Proxable()
end


"""
Proximal operator of the one norm is the soft thresholding operator. 
"""
function prox!(
    ::Proxable, 
    this::OneNormFunction, 
    y::Union{AbstractVecOrMat, Number}, 
    y_out::Union{AbstractVecOrMat, Number},
    lambda::Number, 
    eta::Number
)::Nothing
    for indx in eachindex(y)
        y_out[indx] = abs(y[indx])
        y_out[indx] -= lambda*this.lambda*abs(eta)
        y_out[indx] = max(0, y_out[indx])*sign(y_out[indx])
    end
    return nothing
end


