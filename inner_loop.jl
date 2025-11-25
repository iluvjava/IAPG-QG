# Algorithms for evaluating the proximal operator inexactly, up to a given accuracy. 


"""

This struct is designed to inexactly evaluate proximal problem of the form: 
```
    1/(2λ)‖x - y‖^2 + ω(A*x). 
```

"""
struct InexactProximalPoint
    
    # Fieds that remains constants and shouldn't be changed follow: 
    A::AbstractMatrix
    A_adj::AbstractMatrix
    AA_adj::AbstractMatrix
    "omega as a type of `ClCnvxFxn` must have trait `Proxable`"
    omega::ClCnvxFxn
    """
        Inverted stepsize for ISTA evaluation. Best choice is the spectral norm of AᵀA
    """
    t::Number
    # Fields that mutates now follow.
    z::AbstractVector
    "Intermediate step for Au, for primal objective computations."
    z1::AbstractVector
    v::AbstractVector
    "Intermediate: Aᵀv"
    v1::AbstractVector
    "Intermediate: AAᵀv"
    v2::AbstractVector
    "Intermediate step for prox"
    v3::AbstractVector


    function InexactProximalPoint(
        A::AbstractMatrix, 
        A_adj::AbstractMatrix,
        omega::ClCnvxFxn, 
    )
        # parameter assignments.
        AA_adj = A*A_adj
        (m, n) = size(A)
        t = norm(AA_adj)*1.01
        # memory allocations. 
        z = zeros(m)
        v = zeros(n)
        z1 = similar(z)
        v1 = similar(z)
        v2 = smilar(v)
        v3 = similar(v)
        return new(
            A, A_adj, AA_adj , omega, t, 
            z, z1, v, v1, v2, v3
        )
    end
end


# TODO: Here is a list of things to do for this struct. 
# - [ ]: a function to evaluate the objective value of the primal, given λ. 
# - [ ]: a function to eval the objetive of the dual, given λ. 
# - [ ]: The duality gap for terminations. 
# - [ ]: Given a point and prox problem regularization parameter λ, accuracy ϵ,
#        a point y, it evaluates the proximal. 

"""
Returns the value of : `ω(Az) + 1/(2λ)‖u - v‖²`. 
"""
function eval_primal_objective_at(
    this::InexactProximalPoint, 
    z::AbstractVector,
    y::AbstractVector
)::Number
    ω = this.omega
    A = this.A
    λ = this.lambda
    return ω(A*z) + dot(z - y, z - y)/(2λ)
end


"""
Do one step of primal dual update, using a fixed stepsize which should be valid. 
Update by: 
```
v⁺ = prox[ω⋆](v - (1/τ)(AAᵀv - y))
z⁺ = y - λAᵀv
```

To elimiate garbage collector time, implementations requires the following 
list of intermediate results to be stored after every 
matrix vector multiplication, so we need memory assigned to the heap for 
variables: AAᵀv, Aᵀv, v, z, v⁺, z⁺. 
And we will mutate them. 


"""
function _update_primal_dual!(
    this::InexactProximalPoint, # will mutate. 
    z_out::AbstractVector, # will mutate. 
    v_out::AbstractVector, # will mutate.
    v1::AbstractVector,    # will Mutate. 
    v2::AbstractVector,    # will Mutate. 
    v3::AbstractVector,    # will mutate. 
    v::AbstractVector,     # will reference. 
    y::AbstractVector,     # will reference. 
    lambda::Number
)
    # Referencing. 
    z⁺ = z_out
    v⁺ = v_out
    λ = lambda
    v = v
    Aᵀv = v1
    AᵀAv = v2
    ∇ = v3
    Aᵀ = this.A_adj
    AᵀA = this.AA_adj
    τ = (this.t)*lambda
    ω = this.omega
    # Mutate Aᵀv, AᵀAv 
    mul!(Aᵀv, Aᵀ, v)
    mul!(AᵀAv, AᵀA, v)
    # Mutate ∇ to store: v - (1/τ)(AAᵀv - y)
    ∇ .= AAᵀv
    ∇ .-= y
    ∇ .*= -(1/τ)
    ∇ .+= v
    # Mutae v⁺ by prox of ω at ∇. 
    dprox!(
        ω, 
        v⁺, # mutates
        ∇, 1/τ # no mutate. 
    )
    # Mutate z⁺. 
    z⁺ .= Aᵀv
    z⁺ .*= -λ
    z⁺ .+= y
    return nothing
end



"""
Returns the number of iterations used to achieve the assigned accuracies. 
It mutates the given vectors. 
"""
function do_ista_iteration!(
    this::InexactProximalPoint,
    v_out::AbstractVector, # will mutate
    z_out::AbstractVector, # will mutate
    z0::AbstractVector,    # will reference 
    y::AbstractVector,     # will reference
    lambda::Number, 
    epsilon::Number=1e-6,
    itr_max::Int=8000
)::Number
    # check dimensions of inputs. 
    @assert size(this.v) == size(v_out)
    @assert size(this.z) == size(z_out)
    
    # Referencing assigned resources. 
    λ = lambda
    ϵ = epsilon
    ω = this.omega
    z = this.z
    v = this.v
    A = this.A
    Aᵀv = this.v1
    AAᵀv = this.v2
    Az = this.z1
    z⁺ = z_out  
    v⁺ = v_out
    # Starting the forloop, (z, v) primal dual initial guess. 
    z .= z0
    dprox!(ω, v⁺, v)
    v .= v⁺
    while j <= itr_max
        # update duality gap optimality condition, on (z, v)
        mul!(Az, A, z)
        # TODO: ASSIGN RESOURCES FOR THIS OPERATION AS WELL! 
        p = ω(Az) + dot(z - y, z - y)/(2λ)
        q = (λ/2)*dot(Aᵀv, Aᵀv) - dot(Aᵀv, y) + dval(ω, v)
        if p + q <= ϵ
            # (z⁺, v⁺) from previous iteration exited the for loop. 
            break
        end
        # perform iteration
        j += 1
        _update_primal_dual!(
            this, 
            # will mutate
            z⁺, v⁺, Aᵀv, AAᵀv, this.v3,
            # no mutate
            v, y, λ
        )
        # update reference (z, v) to (z⁺, v⁺)
        z .= z⁺
        v .= v⁺
    end
    

    

    
    
end
