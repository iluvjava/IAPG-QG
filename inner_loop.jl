# Algorithms for evaluating the proximal operator inexactly, up to a given accuracy. 

include("functions_envs.jl")

"""
This struct is designed to inexactly evaluate proximal problem of the form: 
```
    1/(2λ)‖x - y‖^2 + ω(A*x)
```


"""
struct InexacProximalProblem
    A::Function
    omega::ClCnvxFxn

    function InexactProximalProblem(A::Function, omega::ClCnvxFxn)
        return new(A, omega)
    end
end


function InexactProximalProblem(
    A::AbstractMatrix,
    omega::ClCnvxFxn
)::InexactProximalProblem
    return InexacProximalProblem((x) -> A*x, omega)
end


# TODO: Here is a list of things to do for this struct. 
# - [ ]: a function to evaluate the objective value of the primal, given λ. 
# - [ ]: a function to eval the objetive of the dual, given λ. 
# - [ ]: Given a point and prox problem regularization parameter λ, accuracy ϵ,
#        a point y, it evaluates the proximal. 


function primal_objective(
    u::FiniteEuclideanSpace, 
    y::FiniteEuclideanSpace, 
    lambda::Number
)::Number

end

function dual_objective(
    v::FiniteEuclideanSpace, 
    y::FiniteEuclideanSpace, 
    lambda::Number
)::Number

end

function dual_step_update!(
    v_out::FiniteEuclideanSpace, 
    v::FiniteEuclideanSpace, 
    y::FiniteEuclideanSpace, 
    lambda::Number
)::Nothing

end

function primal_step_update!(
    z_out, 
    z, 
    y, 
    lambda
)::Nothing

end

"""
Returns the number of iterations used to achieve the assigned accuracies. 
"""
function do_ista_iteration!(
    z_out, 
    y, 
    lambda, 
    epsilon
)::Number

end
