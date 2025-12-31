# Make function for specifictype of numerical experiment problems. 



"""
Finite Differences, First order, forward, or backward differences, 1D. 

## Options: 
1. `bnd_type::Int=1`: Periodic. 
2. `bnd_type::Int=2`: Mirrored. 
3. `bnd_type::Int=0`: No boundary, truncate it. 


"""
function make_fd_matrix(n::Number, bnd_type::Int=1)::AbstractMatrix
    @assert bnd_type == 0 || bnd_type == 2 || bnd_type == 1
    c = Vector{Int}()
    r = Vector{Int}()
    v = Vector{Float64}()
    h = 1/n
    for i in 0:(n - 1)
        # Diagonal ones
        push!(r, i); push!(c, i); push!(v, h)
        if bnd_type == 0 && i == n - 1
            continue
        end
        push!(r, i)
        if bnd_type == 1
            push!(c, mod(i + 1, n))
        else
            push!(c, i == n - 1 ? n - 2 : i + 1)
        end
        push!(v, -h)
    end
    r .+= 1
    c .+= 1
    return sparse(r, c, v)
end


"""
Finite Differences, second order, central difference at center, 
backwards/forward differences at boundaries. 1D
"""
function make_secord_fd_matrix(n::Number)
    c = Vector{Int}()
    r = Vector{Int}()
    v = Vector{Float64}()
    h = 1/n
    bd = fd = [2, -5, 4, -1]/h
    cd = [1, -2, 1]/h
    # first row, forward differences. 


    # last row, backwards differences. 
    r .+= 1
    c .+= 1
    return sparse(r, c, v)
end

"""
1D convolution using a box kernel. Always periodic. 
The box kernel: [1, ..., 1]/((2*width - 1) + 1)


"""
function box_kernel_averaging(
    n::Number, width::Int=3
)::AbstractMatrix
    l = 2*(width - 1) + 1
    k = 1/l

    r = Vector{Int}()
    c = Vector{Int}()
    v = Vector{Float64}()

    for i = 0:(n - 1)
        for j = (i - div(l, 2)):(i + div(l, 2))
            push!(r, i); push!(c, mod(j, n)); push!(v, k)
        end
    end
    r .+= 1; c .+= 1
    return sparse(r, c, v)
end



