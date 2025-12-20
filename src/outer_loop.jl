
"""
Collect the results produced by the IAPGOuterLoopRunner. 

It has the following
"""
struct ResultsCollector

    j::Vector{Int}
    k::Vector{Int}
    fxn_vals::Vector{Float64}
    pg_norm::Vector{Float64}

    function ResultsCollector()
        return new(Vector{Int}(), Vector{Int}(), Vector{Int}())
    end

end



struct IAPGOuterLoopRunner
    f::ClCnvxFxn                # differentiable. 
    omega::ClCnvxFxn            # Proxable. 
    A::AbstractMatrix           
    ipp::InexactProximalPoint   # inexact prox operator. 
    collector::ResultsCollector # a collector for collecting results. 

    # Constant parameters. 
    p::Number       # error shrinkage power. 
    E::Number       # initial error constant 
    rho::Number     # A simple constant for relative error. 

    # These will mutate. 
    y::Vector{Float64}
    x::Vector{Float64}
    v::Vector{Float64}
    y_next::Vector{Float64}
    x_next::Vector{Float64}
    v_next::Vector{Float64}
    
    # inner loop primal dual iterates. 
    z::Vector{Float64}
    v::Vector{Float64} 

    function IAPGOuterLoopRunner(
        f::ClCnvxFxn, omega::ClCnvxFxn, A::AbstractMatrix;
        p=2, 
        error_scale=1, 
        rho=1
    )
        @assert p > 1 
        @assert error_scale > 0

        return new()
    end


end


### Implementations plans 
### 1. Define all relevant parameters
### 2. Prototype one step of iteration successfully at least. 
### 3. Optimize it while testing it continuously. 

"""
Perform one iteration of inexact proximal gradient method operator and that is. 

### Inputs

These will get mutated: 
- `y⁺`: A temporary, mutating vector for storing `y - (1/L)*∇f(y)`. 
- `y⁺⁺`: Temporary, mutating vector for the results of the 
inexact proximal operator. 
- `v`: Temporary mutating vector for the dual iterates of the inexact proximal
ISTA method. 

These are only referenced: 


"""
function _ipg!(
    this::IAPGOuterLoopRunner,
    y⁺::Vector{Float64},        # Will mutate
    y⁺⁺::Vector{Float64},       # Will mutate
    v::Vector{Float64},         # Will mutate
    ∇fy::Vector{Float64},       # Will reference
    L::Number,                  # Will reference
    ϵ::Number,                 # Will reference
    itr_max::Number=65536       # will reference
)::Number
    ipp = this.ipp
    
    y⁺ .-= (1/(L + ρ))*∇fy
    j = do_ista_iteration!(
        ipp, y⁺⁺, v, y⁺, 
        1/L,                        # ref only
        itr_max=itr_max,            # ref only
        epislon=ϵ
    )
    if j == itr_max
        # inner loop convergence failed. 
        return -1
    end
    return j
end


"""
Performs a step of proximal gradient and then do line search if asked for it. 
This function will be specialized. 


"""
function _ipg_ls!(
    this::IAPGOuterLoopRunner,      
    y⁺::Vector{Float64},            # Will mutate
    y⁺⁺::Vector{Float64},           # Will mutate
    v::Vector{Float64},             # Will mutate
    δy::Vector{Float64},            # Will mutate
    ∇fy::Vector{Float64},           # Will reference
    y::Vector{Float64},             # Will reference
    fy::Float64,                    # will reference
    L::Number,                      # Will reference
    ϵk::Number;                     # Will reference
    ls::Bool=false,                 # Will reference
    lsbtrk::Bool=false              # Will reference
)::Tuple
    
    # Referencing constants of the pg problem. 
    f = this.f
    ρ = this.rho
    # TODO: update `ϵ_k` using ρ over relaxations here! 
    j = _ipg!(
        this, y⁺, y⁺⁺, v, 
        ∇fy,
        L, ρ, ϵk
    )

    if j == -1  # evaluation failed. 
        return -1, L
    end
    
    if ls # armjo line search
        BreakOut = false
        while L < Inf
            δy .= @. y⁺⁺ - y
            BreakOut = f(y⁺⁺) - fy - dot(∇fy, δy) <= (L/2)*dot(δy, δy)
            if BreakOut break end
            if L == Inf error("Line search failed. ") end
            j += _ipg!(
                    this, y⁺, y⁺⁺, v, 
                    ∇fy, L, ρ, ϵk
            )
            L *= 2
            # TODO: update `ϵ_k` and over-relaxation parameter here. 
        end
        
        if lsbtrk
            # backtracking enabled, shrink. 
            L /= 2^(1/1024)
        end
    end

    return j, L
end