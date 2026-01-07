# import LinearAlgebra: adjoint, transpose, mul!, size, norm
import LinearAlgebra
# import Base: * 


abstract type FiniteDifferenceMatrix <: AbstractMatrix{Float64}
end


function LinearAlgebra.norm(this::FiniteDifferenceMatrix, p::Int)::Float64
    @assert p == 2 "Sorry, we only implemented the Frobenius Norm for "*
    "abstract type FiniteDifferenceMatrix"
    return sqrt(this.n*(this.n - 1))
end


### ============================================================================
"""
Implements a first order first derivative forward finite difference 
matrix with periodic period directly. 
"""
struct FastFiniteDiffMatrix <: FiniteDifferenceMatrix
    n::Int
    function FastFiniteDiffMatrix(n::Int) return new(n) end
end


function LinearAlgebra.size(this::FastFiniteDiffMatrix)::Tuple{Int, Int}
return (this.n - 1, this.n) end

function Base.getindex(this::FastFiniteDiffMatrix, i::Int, j::Int)::Float64
    @assert i <= this.n - 1 && j <= this.n
    if j - i == 1 
        return 1.0
    end
    if j - i == 0
        return -1.0
    end
    return 0.0
end

function (Base.:*)(
    this::FastFiniteDiffMatrix, 
    x::AbstractVector{T}
)::AbstractVector{T} where {T<:Number}
    y = zeros(this.n - 1)
    mul!(y, this, x)
    return y
end


function LinearAlgebra.mul!(
    y::AbstractVector{T}, 
    ::FastFiniteDiffMatrix,  
    x::AbstractVector{T}
)::AbstractVector{T} where T <: Number
    # @assert length(y) == length(x) - 1
    @simd for i in eachindex(y)
        y[i] = x[i + 1] - x[i]
    end
    return y
end


"""
Implements the transpose of the first order, first derivative
forward finite difference matrix with periodic period directly. 
"""
struct FastFiniteDiffMatrixTransposed <: FiniteDifferenceMatrix
    n::Int
    function FastFiniteDiffMatrixTransposed(n::Int) return new(n) end
end

function LinearAlgebra.size(this::FastFiniteDiffMatrixTransposed)::Tuple{Int, Int}
return (this.n, this.n - 1) end

function LinearAlgebra.adjoint(::FastFiniteDiffMatrixTransposed)::AbstractMatrix
    return FastFiniteDiffMatrix(this.n)
end

function Base.getindex(this::FastFiniteDiffMatrixTransposed, j::Int, i::Int)
    @assert i <= this.n - 1 && j <= this.n
    if j - i == 1 
        return 1.0
    end
    if j - i == 0
        return -1.0
    end
    return 0.0
end

LinearAlgebra.transpose(
    this::FastFiniteDiffMatrixTransposed
)::AbstractMatrix = adjoint(this)

function LinearAlgebra.adjoint(this::FastFiniteDiffMatrix)::AbstractMatrix
    return FastFiniteDiffMatrixTransposed(this.n)
end
LinearAlgebra.transpose(
    this::FastFiniteDiffMatrix
)::AbstractMatrix = adjoint(this)

function (Base.:*)(
    this::FastFiniteDiffMatrixTransposed, 
    x::AbstractVector{T}
)::AbstractVector{T} where {T <: Number}
    y = zeros(this.n)
    mul!(y, this, x)
    return y
end

function LinearAlgebra.mul!(
    y::AbstractVector{T}, 
    ::FastFiniteDiffMatrixTransposed, 
    x::AbstractVector{T}
)::AbstractVector{T} where T <: Number
    # @assert length(y) == length(x) + 1
    y[1] = -x[1]
    @simd for i in eachindex(y)[2:end - 1]
        y[i] = x[i - 1] - x[i] 
    end
    y[end] = x[end]
    return y
end


