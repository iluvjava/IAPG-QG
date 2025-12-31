using Plots, SparseArrays, ProgressMeter, Statistics, StatsPlots, 
    DataFrames, Latexify, LaTeXStrings

include("../src/import_inner_loop.jl")
include("../src/outer_loop.jl")
include("function_maker.jl")


n = 1024


# Setup the problems
let

    global x = LinRange(-2, 2, n)
    global y = @. ((2pi*x) |> sin |> sign)
    global B = box_kernel_averaging(n, div(n, 50))

    # x: Time domain of the signal. 
    # y: The true signal. 
    # b: Observed signal blurred by A^3 and corrupted. 
    # A: box kernel raised to the power of 3. 
    
    # Corrupt the signal
    global Blurred_Signal = B*y
    global Corrupted_Signal  = [rand() < 0.9 ? NaN : x for x in Blurred_Signal]
    F = downsample_matrix([isnan(i) ? 0.0 : 1.0 for i in Corrupted_Signal])
    global A = F*B
    global b = [isnan(i) ? 0.0 : i for i in Corrupted_Signal]
    

end

# Setup the cost functions of the optimizations problem. 
f = ResidualNormSquared(A, b, 1/n)
ω = OneNormFunction(0.1)
C = make_fd_matrix(n)
rho = 0.1

# Make the outer loop. 
OuterLoop = IAPGOuterLoopRunner(
    f, ω, C, error_scale=1e1, rho=rho, store_fxn_vals=true
)

x0 = ones(n)

@time global Results = run_outerloop_for!(
    OuterLoop, x0, 1e-7, 
    max_itr=4096, lsbtrk=true, show_progress=true,
    inner_loop_settings=InnerLoopSettings(65536, true, 2048)
)


# Observed VS The denoised signal. 
# x: The grid. 
# y: The original signal 
# Results.x: The deblurred signal.  
p1 = scatter(
    x, Corrupted_Signal, 
    title="Corrupted Signal VS The Recovered Signal", 
    color=:gray, 
    label="Corrupted Signal", 
    marker=:x, 
    markerstrokewidth=3, 
    markersize=5, 
    size=(800, 600), 
    dpi=330
)
plot!(
    p1, x, Results.x, 
    color=:blue, alpha=0.5, linewidth=3, label="Recovered"
)
p1|>display

# Denoised VS Original Signal
p2 = scatter(
    x, y, 
    title="Ground Truth VS The Recovered Signal", 
    color=:gray, 
    label="Ground Truth", 
    marker=:x, 
    markerstrokewidth=3, 
    markersize=5, 
    size=(800, 600), 
    dpi=330
)
plot!(
    p2, x, Results.x, 
    color=:blue, alpha=0.5, linewidth=3, label="Recovered"
)
p2 |> display

# Cumulative iterations of inner loop VS outer loop

InnerLoop_ItrJ = Results.j[1:end - 1] # prevent last one is -1.
ks = 1:(length(Results.j) - 1 )
p3 = plot(
    ks, 
    InnerLoop_ItrJ,
    title="Log2 Inner Loop Iterations for each\n outer Loop Iteration",
    label="Inner Loop Iterations"
)
plot!(
    p3, ks, 
    3000*log2.(ks), 
    label="1.2*log2(k) for reference"
)
p3 |> display

# Total inner loop iterations vs log2(‖xk-yk‖)



# Relative + Absolute Tolerance 
# vs Inner Loop Iteration per Outer Loop Iterations
# With a Log Log plot

Epsilons = Results.epsilon
Relative_Errors = @. Results.dy*rho*Results.ss
TotalErrors = @. Epsilons + Relative_Errors


