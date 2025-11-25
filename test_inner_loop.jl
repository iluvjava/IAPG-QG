using Test

@testset "‖⋅‖_1 Function Basic Tests" begin
    include("import_inner_loop.jl")

    # Test subjects and function now follows. 
    λ = 1.1
    n = 3
    f = OneNormFunction(λ) 
    y = randn(n)

    function test_function_value()
        @info "Testing function value evaluation. "
        z = f(y)
        @assert isapprox(z, λ*norm(y, 1))
        return true
    end

    function test_resolvent_identity()
        @info "Testing the basic case of the Resolvent identity of proximal operator. "
        zp = similar(y)
        zd = similar(y)
        dprox!(f, zd, y, 1) # zd mutated 
        prox!(f, zp, y, 1)
        @assert isapprox(zp + zd, y)
        return true
    end

    function dval_dprox()
        @info "Testing if dprox returns dual feasible value.  "
        zd = similar(y)
        dprox!(f, zd, y, 1) # zd mutated 
        d_val = dval(f, zd)

        return d_val < Inf64
    end

    @test test_function_value()
    @test test_resolvent_identity()
    @test dval_dprox()

end


@testset "Testing the inner loop that solves the proximal point problem" begin

    A = randn(3, 4)
    ω = OneNormFunction(0.1)


end