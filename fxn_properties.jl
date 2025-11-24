### ============================================================================
### FUNCTION OPERATIONS
### ============================================================================


const FiniteEuclideanSpace = Union{AbstractVecOrMat, Number}


abstract type ClCnvxFxn
    # closed convex function, prox it, or gradient it. 
    # If some of them are not proxable, or differentible, it throws an error 
    # when prox!, and grad_and_fxnval! is used. 
end


function (this::ClCnvxFxn)(::FiniteEuclideanSpace)::Number
    error("(::ClCnvxFxn)(x) not implemented for type `$this` yet. ")
end

function differentiable_trait_assigner(this::ClCnvxFxn)::TraitsOfClCnvxFxn
    # IMPLEMENT THIS FOR SPECIFIC TYPES! 
    error("Differentiable traits unassigned for type `$this` yet. ")
end

function prox_trait_assign(this::ClCnvxFxn)::TraitsOfClCnvxFxn
    # IMPLEMENT THIS FOR SPECIFIC TYPES! 
    error("Proxable trait unassigned for type `$this` yet. ")
end

function dval_trait_assigner(this::ClCnvxFxn)::TraitsOfClCnvxFxn
    error(
        "HasFenchelDual trait assigner"*
        " not implemented for $(typeof(this)) yet. "
    )
end

function grad_and_fxnval!(
    this::ClCnvxFxn, 
    x::FiniteEuclideanSpace,
    x_out::FiniteEuclideanSpace
)::Number
# IMPLEMENT THIS FOR SPECIFIC TYPES! 
    return grad_and_fxnval!(
        differentiable_trait_assigner(this), 
        this, x, x_out
    ) 
end

function prox!(
    this::ClCnvxFxn, 
    y::FiniteEuclideanSpace, 
    y_out::FiniteEuclideanSpace, 
    lambda::Number = 1, 
    eta::Number = 1
)::Nothing
    # IMPLEMENT THIS FOR SPECIFIC TYPES! 
    return prox!(
        prox_trait_assign(this), this, 
        y, y_out, lambda, eta
    )
end

"""
Evaluates the proximal operator of the dual of (x |> lambda*f(x)) at y. 
"""
function dprox!(
    this::ClCnvxFxn, 
    y::FiniteEuclideanSpace, 
    y_out::FiniteEuclideanSpace, 
    lambda::Number=1
)::Nothing
    error("Not yet implemented. ")
    prox!(this, y, y_out, lambda)
    # TODO IMPLEMENT THIS. 
end

function dval(
    this::ClCnvxFxn, 
    x::FiniteEuclideanSpace
)
    dval(dval_trait_assigner(this), this, x)
end


### ============================================================================
### FUNCTION TRAITS
### ============================================================================
### Impelemetations of operations on ClCnvxFxn varies according to different 
### type of traits the function as assigned to has, or don't. 

abstract type TraitsOfClCnvxFxn
    
end


struct Proxable <: TraitsOfClCnvxFxn
    # Function has proximal operator. 
end

"""
Proximal operator of the function (x |-> lambda*f(eta*x)) at y
"""
function prox!(
    ::Proxable,
    this::ClCnvxFxn, 
    y::FiniteEuclideanSpace, 
    y_out::FiniteEuclideanSpace,
    lambda::Number, 
    eta::Number
)::Nothing
    error("Function `prox` not yet implemented for $(typeof(this))")
end


### ============================================================================
struct Differentiable <: TraitsOfClCnvxFxn
    # Function has a gradient. 
end

function grad_and_fxnval!(
    ::Differentiable,
    this::ClCnvxFxn, 
    x::FiniteEuclideanSpace,
    x_out::FiniteEuclideanSpace
)::Number
    error("Function `grad_and_fxnval` not yet implemented for $(typeof(this))")
    # return the function value! 
end

### ============================================================================

struct HasFenchelDual <: TraitsOfClCnvxFxn
    # Knowing the primal of the function allows us to evaluate the dual of the 
    # function as well. 
end

"""
Evalute the dual of this current ClCnvxFxn at point `x`. 
"""
function dval(
    ::HasFenchelDual, 
    this::ClCnvxFxn, 
    x::FiniteEuclideanSpace
)::Number
    error("Function `dval` not implemented by type $(typeof(this)) yet. ")
end