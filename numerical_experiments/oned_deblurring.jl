using Plots, SparseArrays, ProgressMeter, Statistics, StatsPlots, 
    DataFrames, Latexify, LaTeXStrings

include("../src/import_inner_loop.jl")
include("../src/outer_loop.jl")
include("function_maker.jl")


n = 1024


# Setup the problems
let

    x = LinRange(-2, 2, n)    
    global y = @. ((2pi*x) |> sin |> sign)
    global A = box_kernel_averaging(n, div(n, 100))
    A = A^3
    # y: The true signal. 
    # b: Observed signal blurred by A^3 and corrupted. 
    # A: box kernel raised to the power of 3. 
    global b = A*y
    b .+= 1e-1*randn(n)
end

# Setup the cost functions of the optimizations problem. 
f = ResidualNormSquared(A, b)
ω = OneNormFunction(0.1)

# Make the outer loop. 

