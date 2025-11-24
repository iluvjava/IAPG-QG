# Algorithms for evaluating the proximal operator inexactly, up to a given accuracy. 

include("functions_envs.jl")


"""

This struct is designed to inexactly evaluate proximal problem of the form: 
```
    1/(2λ)‖x - y‖^2 + ω(A*x). 
```

"""
struct InexacProximalProblem
    
    A::AbstractMatrix
    A_adj::AbstractMatrix
    AA_adj::AbstractMatrix
    "omega as a type of `ClCnvxFxn` must have trait `Proxable`"
    omega::ClCnvxFxn
    """
        Inverted stepsize for ISTA evaluation. Best choice is the spectral norm of AᵀA
    """
    tau::Number

    # Maybe: z, v, z_next, v_next, av, v_temp


    function InexacProximalProblem(
        A::AbstractMatrix, 
        A_adj::AbstractMatrix,
        omega::ClCnvxFxn, 
        tau::Number
    )
        if tau <= 0
            error(
                "Tau in `InexacProximalProblem` cannot"
                *" be <= 0 but here we have tau=$tau. "
            )
        end
        return new(A, A_adj, A*A_adj , omega, tau)
    end
end


# TODO: Here is a list of things to do for this struct. 
# - [ ]: a function to evaluate the objective value of the primal, given λ. 
# - [ ]: a function to eval the objetive of the dual, given λ. 
# - [ ]: The duality gap for terminations. 
# - [ ]: Given a point and prox problem regularization parameter λ, accuracy ϵ,
#        a point y, it evaluates the proximal. 


"""
ω(Au) + 1/(2λ)‖u - y‖^2
"""
function primal_objective(
    this::InexacProximalProblem,
    u::FiniteEuclideanSpace, 
    au::FiniteEuclideanSpace,
    y::FiniteEuclideanSpace, 
    lambda::Number
)::Number
    ω = this.omega
    Au = au
    λ = lambda
    return ω(Au) + dot(u - y, u - y)/(2λ)
end

"""
(λ/2)‖Aᵀv‖² - ⟨Aᵀv,y⟩ + [ω⋆](v)
"""
function dual_objective(
    ::InexacProximalProblem,
    v::FiniteEuclideanSpace,
    av::FiniteEuclideanSpace, 
    y::FiniteEuclideanSpace, 
    lambda::Number
)::Number
    λ = lambda
    Aᵀv = av
    ω = omega
    return (λ/2)*dot(Aᵀv, Aᵀv) - dot(Aᵀv, y) + dval(ω, v)
end


"""
The duality gap here follows the paper. 
One wants the convergence of the sequence of (z, v). 
"""
function duality_gap(
    this::InexacProximalProblem,
    z::FiniteEuclideanSpace, 
    az::FiniteEuclideanSpace,
    v::FiniteEuclideanSpace, 
    av::FiniteEuclideanSpace,
    y::FiniteEuclideanSpace, 
    lambda::Number
)::Number
    Aᵀv = av
    Az = az
    p = primal_objective(
        this, z, Az, y, lambda
    )
    q = dual_objective(
        this, v, Aᵀv, y, lambda 
    )
    return p + q
end


"""
Do one step of primal dual update, together with line search and checks. 
It basically performs a step of proximal gradient. 
The Bregman Divergence of the smooth parts of the dual is given by: 

d(x, y) ↦ λ⟨Aᵀ(x - y), Aᵀ(x - y)⟩/2


update by: 
```
v⁺ = prox[ω⋆](v - (1/τ)(AAᵀv - y))
z⁺ = y - λAᵀv
```

"""
function dual_proximal_gradient!(
    this::InexacProximalProblem,
    z_out::FiniteEuclideanSpace,
    v_out::FiniteEuclideanSpace,
    z::FiniteEuclideanSpace,
    v::FiniteEuclideanSpace,
    av::FiniteEuclideanSpace,
    y::FiniteEuclideanSpace, 
    lambda::Number
)
    

end



"""
Returns the number of iterations used to achieve the assigned accuracies. 
It mutates the given vectors. 
"""
function do_ista_iteration!(
    this::InexacProximalProblem,
    v_out::FiniteEuclideanSpace,
    z_out::FiniteEuclideanSpace, 
    y::FiniteEuclideanSpace, 
    lambda::Number, 
    epsilon::Number
)::Number

    
end
