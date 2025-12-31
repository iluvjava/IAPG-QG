using Test
using UnicodePlots
using SparseArrays
include("import_inner_loop.jl")
include("outer_loop.jl")

@testset "Testing Outer Loop" begin 
    
    function instantiation(
    )::Tuple{IAPGOuterLoopRunner, Vector{Float64}}
        m = 1024
        n = 1024
        l = 1024
        A = diagm(ones(n))
        b = zeros(m)
        C = randn(l, n)
        f = ResidualNormSquared(C, b)
        ω = OneNormFunction(0.01)
        OuterLoop = IAPGOuterLoopRunner(f, ω, A)

        return OuterLoop, randn(n)

    end
    
    function test_instantioation()::Bool
        @info "Testing Instantiation. "
        OuterLoop, x = instantiation()
        return true
    end

    function test_iterations()::Bool
        @info "Test if iteration go ok for the outerloop on simple problem. "
        # Make the test instance 
        m = 128
        n = 128
        l = 128
        A = Diagonal(randn(n))  # Sparse Matrix here. 
        b = zeros(m)
        C = randn(l, n)
        f = ResidualNormSquared(C, b)
        ω = OneNormFunction(1)
        OuterLoop = IAPGOuterLoopRunner(
            f, ω, A, error_scale=3000, rho=1, store_fxn_vals=true
        )
        x0 = ones(n)
        global Results = run_outerloop_for!(
            OuterLoop, x0, 1e-10, max_itr=65536, 
            inner_loop_settings=InnerLoopSettings(4096, true, 2048)
        )
        @info "Reporting Results. "
        lineplot(
            1:length(Results.dy), 
            Results.dy.|>log2, 
            title="log2(‖x_k - y_k‖)"
        )|>println
        lineplot(
            1:length(Results.j), 
            Results.j,
            title="Inner total iterations vs each outer iteration"
        )|>println
        return true
    end

    function comparing_iterations()::Bool
        @info "Simple, TVL1 signal recovery with, smoothing and pepper noise. "
        # Setup parameters for the problems: A, C, b. 
        
    end

    

    @test test_instantioation()
    @test test_iterations()

end