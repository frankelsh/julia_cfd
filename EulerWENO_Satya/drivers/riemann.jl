
using Revise
using EulerWENO, Plots

function run_riemann(T::DataType) 
    e = Euler{T}(nx=100, cfl=0.1)

    # Generate Grid
    x  = range(e.xl+e.dx/2, length=e.nx, step=e.dx)
    y  = range(e.yl+e.dy/2, length=e.ny, step=e.dy)
    icase = 3
    # Initialize primitive variables
    init2Driemann(e, icase)
    # Initial solution vector
    u = zeros(T, e.nx, e.ny, 4)
    solvec(e, u)

    dt = timestep(e)

    Tfinal = T(0.3)
    nsteps = ceil(Int, Tfinal/dt)
    # nsteps = 10

    println("Case ", icase)
    t = T(0)
    @time for i in 1:nsteps
        rk3tvdmp5(e, u)
        t = t + dt
        println("Time is ", t, "  Max pressure is ", maximum(e.p))
    end
    println("Plot me!")
    contourf(x, y, transpose(e.p), xlabel="x", ylabel="y", legend=true, title="Pressure", fmt = :png)
end

run_riemann(Float64)