module EulerWENO

using StaticArrays

export Euler, init2Driemann, init2DSchulzRinne, solvec, decomp, timestep, rk3tvdweno5, rk3tvdmp5

const gam = 1.4
const gm1 = gam - 1.0
const gm1i = 1.0/gm1

const nv = 4

const pg = [3/10, 3/5, 1/10]
const mg = [1/10, 3/5, 3/10]

Base.@kwdef mutable struct Euler{T<:AbstractFloat}
    nx::Int
    ny::Int = nx
    cfl::T
    rho::Array{T,2} = zeros(nx, ny) # density, shape (nx, ny)
    ux::Array{T,2} = zeros(nx, ny) # x-velocity, shape (nx, ny)
    uy::Array{T,2} = zeros(nx, ny) # y-velocity, shape (nx, ny)
    p::Array{T,2} = zeros(nx, ny) # pressure, shape (nx, ny)
    E::Array{T,2} = zeros(nx, ny) # energy, shape (nx, ny)
    c::Array{T,2} = zeros(nx, ny) # sound speed, shape (nx, ny)
    H::Array{T,2} = zeros(nx, ny) # enthalpy, shape (nx, ny)
    f::Array{T,3} = zeros(nx, ny, nv) # x-flux, shape (nx, ny, 4)
    g::Array{T,3} = zeros(nx, ny, nv) # y-flux, shape (nx, ny, 4)
    # Time stepping
    u1::Array{T,3} = zeros(nx, ny, nv)
    u2::Array{T,3} = zeros(nx, ny, nv)
    # RHS evaluation
    b::Array{T,3} = zeros(nx, ny, nv)
    fh::Array{T,3} = zeros(nx, ny, nv)
    gh::Array{T,3} = zeros(nx, ny, nv)
    # Domain bounds
    xl::T = T(0)
    xr::T = T(1)
    yl::T = T(0)
    yr::T = T(1)
    # Grid step size
    dx::T = (xr-xl)/nx
    dy::T = (yr-yl)/ny
end

function init2DSchulzRinne(e::Euler{T}) where T<:Real
    # Case 3 from Liksa and Wendroff paper
    ulb = T(4/sqrt(11)); vlb = T(4/sqrt(11)); rholb = T(77/558);  plb = T(9/310)
    ult = T(4/sqrt(11)); vlt = T(0);   rholt = T(33/62); plt = T(0.3)
    urb = T(0);   vrb = T(4/sqrt(11)); rhorb = T(33/62); prb = T(0.3)
    urt = T(0);   vrt = T(0);   rhort = T(1.5);    prt = T(1.5)
    io2 = div(e.nx,T(1/0.8))
    jo2 = div(e.ny,T(1/0.8))
    for j in 1:e.ny
        for i in 1:e.nx
            if i <= io2 && j <= jo2
                e.ux[i,j]  = ulb
                e.uy[i,j]  = vlb
                e.rho[i,j] = rholb
                e.p[i,j]   = plb
            elseif i <= io2 && j > jo2
                e.ux[i,j]  = ult
                e.uy[i,j]  = vlt
                e.rho[i,j] = rholt
                e.p[i,j]   = plt
            elseif i > io2 && j <= jo2
                e.ux[i,j]  = urb
                e.uy[i,j]  = vrb
                e.rho[i,j] = rhorb
                e.p[i,j]   = prb
            elseif i > io2 && j > jo2
                e.ux[i,j]  = urt
                e.uy[i,j]  = vrt
                e.rho[i,j] = rhort
                e.p[i,j]   = prt
            end
            ke = T(0.5)*(e.ux[i,j]^2+e.uy[i,j]^2)
            e.E[i,j] = gm1i*e.p[i,j]/e.rho[i,j] + ke
            e.c[i,j] = sqrt(gam*e.p[i,j]/e.rho[i,j])
            e.H[i,j] = e.E[i,j] + e.p[i,j]/e.rho[i,j]
        end
    end
end

function init2Driemann(e::Euler{T}, icase) where T<:Real
    if icase == 3
        # Case 3 from Liksa and Wendroff paper
        ulb = T(1.206); vlb = T(1.206); rholb = T(0.138);  plb = T(0.029)
        ult = T(1.206); vlt = T(0);   rholt = T(0.5323); plt = T(0.3)
        urb = T(0);   vrb = T(1.206); rhorb = T(0.5323); prb = T(0.3)
        urt = T(0);   vrt = T(0);   rhort = T(1.5);    prt = T(1.5)
    elseif icase == 4
        # Case 12 from Liksa and Wendroff paper
        ulb = T(0.8939); vlb = T(0.8939); rholb = T(1.1);  plb = T(1.1)
        ult = T(0.8939); vlt = T(0);   rholt = T(0.5065); plt = T(0.35)
        urb = T(0);   vrb = T(0.8939); rhorb = T(0.5065); prb = T(0.35)
        urt = T(0);   vrt = T(0);   rhort = T(1.1);    prt = T(1.1)
    elseif icase == 5
        # Case  from Liksa and Wendroff paper
        ulb = T(0.75); vlb = T(0.5); rholb = T(1);  plb = T(1)
        ult = T(-0.75); vlt = T(0.5);   rholt = T(2); plt = T(1)
        urb = T(0.75);   vrb = T(-0.5); rhorb = T(3); prb = T(1)
        urt = T(-0.75);   vrt = T(0.5);   rhort = T(1);    prt = T(1)
    elseif icase == 6
        # Case  from Liksa and Wendroff paper
        ulb = T(0.75); vlb = T(0.5); rholb = T(1);  plb = T(1)
        ult = T(0.75); vlt = T(0.5);   rholt = T(2); plt = T(1)
        urb = T(-0.75);   vrb = T(-0.5); rhorb = T(3); prb = T(1)
        urt = T(0.75);   vrt = T(-0.5);   rhort = T(1);    prt = T(1)
    elseif icase == 12
        # Case 12 from Liksa and Wendroff paper
        ulb = T(0); vlb = T(0); rholb = T(0.8);  plb = T(1)
        ult = T(0.75); vlt = T(0.5);   rholt = T(2); plt = T(1)
        urb = T(0);   vrb = T(0.7276); rhorb = T(1); prb = T(1)
        urt = T(0);   vrt = T(0);   rhort = T(0.5313);    prt = T(0.4)
    else
        error("Unknown Riemann problem")
    end

    io2 = div(e.nx,2)
    jo2 = div(e.ny,2)
    for j in 1:e.ny
        for i in 1:e.nx
            if i <= io2 && j <= jo2
                e.ux[i,j]  = ulb
                e.uy[i,j]  = vlb
                e.rho[i,j] = rholb
                e.p[i,j]   = plb
            elseif i <= io2 && j > jo2
                e.ux[i,j]  = ult
                e.uy[i,j]  = vlt
                e.rho[i,j] = rholt
                e.p[i,j]   = plt
            elseif i > io2 && j <= jo2
                e.ux[i,j]  = urb
                e.uy[i,j]  = vrb
                e.rho[i,j] = rhorb
                e.p[i,j]   = prb
            elseif i > io2 && j > jo2
                e.ux[i,j]  = urt
                e.uy[i,j]  = vrt
                e.rho[i,j] = rhort
                e.p[i,j]   = prt
            end
            ke = T(0.5)*(e.ux[i,j]^2+e.uy[i,j]^2)
            e.E[i,j] = gm1i*e.p[i,j]/e.rho[i,j] + ke
            e.c[i,j] = sqrt(gam*e.p[i,j]/e.rho[i,j])
            e.H[i,j] = e.E[i,j] + e.p[i,j]/e.rho[i,j]
        end
    end
end

function decomp(e::Euler{T}, u::AbstractArray{T}) where T<:Real
    @inbounds for j=1:e.ny
       @inbounds @fastmath @simd for i=1:e.nx
            e.rho[i,j] = u[i,j,1]
            e.ux[i,j]  = u[i,j,2]/u[i,j,1]
            e.uy[i,j]  = u[i,j,3]/u[i,j,1]
            e.E[i,j]   = u[i,j,4]/u[i,j,1]
            ke         = T(0.5)*(e.ux[i,j]^2+e.uy[i,j]^2)
            e.p[i,j]   = gm1*e.rho[i,j]*(e.E[i,j] - ke)
            # if e.p[i,j] < 0.0f0
            #     e.p[i,j] = 0.001f0
            #     #println("Negative pressure at ", i, " ", j)
            # end
            #c[i,j]   = sqrt(gam*abs(p[i,j])/abs(rho[i,j]))
            e.c[i,j]   = sqrt(gam*e.p[i,j]/e.rho[i,j])
            e.H[i,j]   = e.E[i,j] + e.p[i,j]/e.rho[i,j]
        end
    end
end

function solvec(e::Euler{T}, u::AbstractArray{T}) where T<:Real
    @inbounds for j=1:e.ny
        @inbounds for i=1:e.nx
            u[i,j,1] = e.rho[i,j]
            u[i,j,2] = e.rho[i,j]*e.ux[i,j]
            u[i,j,3] = e.rho[i,j]*e.uy[i,j]
            u[i,j,4] = e.rho[i,j]*e.E[i,j]
        end
    end
end

# function fluxvecx(e::Euler)
#     for j=1:e.ny
#         for i=1:e.nx
#             ke = 0.5*(e.ux[i,j]^2+e.uy[i,j]^2)
#             p  = gm1*e.rho[i,j]*(e.E[i,j]-ke)
#             e.f[i,j,1] = e.rho[i,j]*e.ux[i,j]
#             e.f[i,j,2] = e.rho[i,j]*e.ux[i,j]^2+p
#             e.f[i,j,3] = e.rho[i,j]*e.ux[i,j]*e.uy[i,j]
#             e.f[i,j,4] = e.ux[i,j]*(e.rho[i,j]*e.E[i,j] + p)
#         end
#     end
# end

# function fluxvecy(e::Euler)
#     for j=1:ny
#         for i=1:nx
#             ke = 0.5*(e.ux[i,j]^2+e.uy[i,j]^2)
#             p  = gm1*e.rho[i,j]*(e.E[i,j]-ke)
#             e.g[i,j,1] = e.rho[i,j]*e.uy[i,j]
#             e.g[i,j,2] = e.rho[i,j]*e.ux[i,j]*e.uy[i,j]
#             e.g[i,j,3] = e.rho[i,j]*e.uy[i,j]^2+p
#             e.g[i,j,4] = e.uy[i,j]*(e.rho[i,j]*e.E[i,j] + p)
#         end
#     end
# end

function riemann_solvers(n1::T,n2::T,qL::AbstractVector{T},qR::AbstractVector{T}) where T<:Real
    # n1 = 1.0, n2 = 0.0 for x
    # n1 = 0.0, n2 = 1.0 for y

    left_eigv  = @MMatrix zeros(T,nv,nv)
    right_eigv = @MMatrix zeros(T,nv,nv)
    fL    = @MVector zeros(T,nv)
    fR    = @MVector zeros(T,nv)
    del   = @MVector zeros(T,nv)
    alpha = @MVector zeros(T,nv)
    fiph  = @MVector zeros(T,nv)

    #Primitives
    rhoL = qL[1]
    uL   = qL[2]/qL[1]
    vL   = qL[3]/qL[1]
    EL   = qL[4]/qL[1]
    pL   = gm1*rhoL*(EL - T(0.5)*(uL^2+vL^2))
    # if pL < 0.0
    #     pL = 0.001
    # end
    cL   = sqrt(gam*pL/rhoL)
    HL   = EL + pL/rhoL

    rhoR = qR[1]
    uR   = qR[2]/qR[1]
    vR   = qR[3]/qR[1]
    ER   = qR[4]/qR[1]
    pR   = gm1*rhoR*(ER - T(0.5)*(uR^2+vR^2))
    # if pR < 0.0
    #     pR = 0.001
    # end
    cR   = sqrt(gam*pR/rhoR)
    HR   = ER + pR/rhoR
    
    # Fluxes 
    fL[1] = n1*rhoL*uL + n2*rhoL*vL
    fL[2] = n1*(rhoL*uL^2 + pL) + n2*rhoL*uL*vL
    fL[3] = n1*rhoL*uL*vL + n2*(rhoL*vL^2 + pL)
    fL[4] = n1*rhoL*uL*HL + n2*rhoL*vL*HL

    fR[1] = n1*rhoR*uR + n2*rhoR*vR
    fR[2] = n1*(rhoR*uR^2 + pR) + n2*rhoR*uR*vR
    fR[3] = n1*rhoR*uR*vR + n2*(rhoR*vR^2 + pR)
    fR[4] = n1*rhoR*uR*HR + n2*rhoR*vR*HR

    rhodenom = (sqrt(rhoL) + sqrt(rhoR))
    rhohat = sqrt(rhoL*rhoR)
    uhat = (sqrt(rhoL)*uL + sqrt(rhoR)*uR)/ rhodenom
    vhat = (sqrt(rhoL)*vL + sqrt(rhoR)*vR)/ rhodenom
    Hhat = (sqrt(rhoL)*HL + sqrt(rhoR)*HR)/ rhodenom
    arg = gm1*(Hhat - T(0.5)*(uhat^2+vhat^2))
    # if arg < 0.0
    #     arg = 0.001
    # end    
    chat = sqrt(arg)
    qn = n1*uhat + n2*vhat
    qt = n1*uL + n2*uR
    
    iriemann = 1

    if iriemann == 1 #ihll == 1
        SL   = min(qt-cL,qn-chat)
        SR   = max(qt+cR,qn+chat)

        if T(0) <= SL
            @inbounds @fastmath for l in 1:nv
                fiph[l] = fL[l]
            end
        elseif SL <= T(0) && SR >= T(0)
            @inbounds @fastmath for l in 1:nv
                fiph[l] = (SR*fL[l] - SL*fR[l] + SL*SR*(qR[l] - qL[l]))/(SR-SL)
            end
        elseif T(0) >= SR
            @inbounds @fastmath for l in 1:nv
                fiph[l] = fR[l]
            end
        end
        
        return SVector(fiph)
    
    elseif iriemann == 2 #ihllc == 1
        SL   = min(qt-cL,qn-chat)
        SR   = max(qt+cR,qn+chat)
        
        Sstr  = (pR - pL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR))/(rhoL*(SL-uL)-rhoR*(SR-uR))
        UstrL= (SL-uL)/(SL-Sstr)*[rhoL, rhoL*Sstr, rhoL*vL, EL+(Sstr-uL)*(rhoL*Sstr+pL/(SL-uL))]
        UstrR= (SR-uR)/(SR-Sstr)*[rhoR, rhoR*Sstr, rhoR*vR, ER+(Sstr-uR)*(rhoR*Sstr+pR/(SR-uR))]
        
        if SL >= T(0)
            @inbounds @fastmath for l in 1:nv
                fiph[l] = fL[l]
            end
        elseif SL < T(0) && Sstr >= T(0)
            @inbounds @fastmath for l in 1:nv
                fiph[l] = fL[l] + SL*(UstrL[l] - qL[l])
            end 
        elseif SR > T(0) && Sstr <= T(0)
            @inbounds @fastmath for l in 1:nv
                fiph[l] = fR[l] + SR*(UstrR[l] - qR[l])
            end 
        elseif SR <= T(0)
            @inbounds @fastmath for l in 1:nv
                fiph[l] = fR[l]
            end
        end
        
        return SVector(fiph)
    
    elseif iriemann == 3 #ihllclm == 1
        SL   = min(qt-cL,qn-chat)
        SR   = max(qt+cR,qn+chat)
        
        Sstr  = (pR - pL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR))/(rhoL*(SL-uL)-rhoR*(SR-uR))
        UstrL= (SL-uL)/(SL-Sstr)*[rhoL, rhoL*Sstr, rhoL*vL, EL+(Sstr-uL)*(rhoL*Sstr+pL/(SL-uL))]
        UstrR= (SR-uR)/(SR-Sstr)*[rhoR, rhoR*Sstr, rhoR*vR, ER+(Sstr-uR)*(rhoR*Sstr+pR/(SR-uR))]
        
        Malimit = T(0.1)
        Malocal = max(abs(uL/cL),abs(uR/cR))
        phi  = sin(min(T(1),Malocal/Malimit)*pi*T(0.5))
        SL   = phi*SL
        SR   = phi*SR
        if SL >= T(0)
            @inbounds @fastmath for l in 1:nv
                fiph[l] = fL[l]
            end 
        elseif SL < T(0) && Sstr >= T(0)
            @inbounds @fastmath for l in 1:nv
                fiph[l] = fL[l] + SL*(UstrL[l] - qL[l])
            end
        elseif SR > T(0) && Sstr <= T(0)
            @inbounds @fastmath for l in 1:nv
                fiph[l] = fR[l] + SR*(UstrR[l] - qR[l])
            end 
        elseif SR <= T(0)
            @inbounds @fastmath for l in 1:nv
                fiph[l] = fR[l]
            end
        end
        
        return SVector(fiph)
            
    elseif iriemann == 4 # Roe
        delrho  = rhoR - rhoL
        delrhou = rhoR*uR - rhoL*uL
        delrhov = rhoR*vR - rhoL*vR
        delrhoE = rhoR*ER - rhoL*EL
        del = [delrho, delrhou, delrhov, delrhoE]

        # Eigenvalues
        lam1 = abs(qn - chat)
        lam2 = abs(qn) 
        lam3 = abs(qn)
        lam4 = abs(qn+chat) 

        # Sonic correction
        if lam1 < T(0.5)
            lam1 = T(0.5)*(T(0.5) + lam1^2/T(0.5))
        end
        if lam4 < T(0.5)
            lam4 = T(0.5)*(T(0.5) + lam4^2/T(0.5))
        end

        # Wave amplitudes
        lev,rev = eigenvec(nx,ny,uhat,vhat,chat,Hhat)
        alpha = lev*del

        @inbounds @fastmath for i in 1:nv
            fiph[i] = T(0.5)*(fL[i] + fR[i]) - T(0.5)*(lam1*alpha[1]*rev[i,1] + 
                                                lam2*alpha[2]*rev[i,2] + 
                                                lam3*alpha[3]*rev[i,3] +
                                                lam4*alpha[4]*rev[i,4])
            #fiph[i] = 0.5*(fL[i] + fR[i]) - 0.5*max(lam1,lam2,lam3,lam4)*del[i]

        end

        return SVector(fiph) 

    end
end

function localfluxx(uu)
    ke = 0.5*(uu[2]^2+uu[3]^2)/uu[1]^2
    pp = gm1*(uu[4] - uu[1]*ke)
    @SVector [
        uu[2],
        uu[2]^2/uu[1] + pp,
        uu[2]*uu[3]/uu[1],
        uu[2]/uu[1]*(uu[4] + pp)
    ]
end

function localfluxy(uu)
    ke = 0.5*(uu[2]^2+uu[3]^2)/uu[1]^2
    pp = gm1*(uu[4] - uu[1]*ke)
    @SVector [
        uu[3],
        uu[2]*uu[3]/uu[1],
        uu[3]^2/uu[1] + pp,
        uu[3]/uu[1]*(uu[4] + pp)
    ]
end

@inline function eigenvec(nx::Int,ny::Int,u::T,v::T,c::T,H::T) where T<:Real
    # nx = 1.0, ny = 0.0 for x
    # nx = 0.0, ny = 1.0 for y

    # Following Hiro I Do Like CFD
    # Eqns. 3.5.8-3.14
    left_eigv = @MMatrix zeros(T,nv,nv)
    right_eigv = @MMatrix zeros(T,nv,nv)

    q2 = u^2+v^2
    qn = u*nx + v*ny
    lx = -ny
    ly = nx
    ql = u*lx + v*ly

    left_eigv[1,1] = T(0.5)*(T(0.5)*gm1/c^2*q2 + qn/c)
    left_eigv[1,2] = -T(0.5)*(gm1/c^2*u + nx/c)
    left_eigv[1,3] = -T(0.5)*(gm1/c^2*v + ny/c)
    left_eigv[1,4] = gm1*T(0.5)/c^2

    left_eigv[2,1] = -ql
    left_eigv[2,2] = lx
    left_eigv[2,3] = ly
    left_eigv[2,4] = T(0)

    left_eigv[3,1] = T(1) - T(0.5)*gm1/c^2*q2
    left_eigv[3,2] = gm1/c^2*u
    left_eigv[3,3] = gm1/c^2*v
    left_eigv[3,4] = -gm1/c^2

    left_eigv[4,1] = T(0.5)*(T(0.5)*gm1/c^2*q2 - qn/c)
    left_eigv[4,2] = -T(0.5)*(gm1/c^2*u - nx/c)
    left_eigv[4,3] = -T(0.5)*(gm1/c^2*v - ny/c)
    left_eigv[4,4] = gm1*T(0.5)/c^2

    right_eigv[1,1] = T(1)
    right_eigv[1,2] = T(0)
    right_eigv[1,3] = T(1)
    right_eigv[1,4] = T(1)

    right_eigv[2,1] = u - c*nx
    right_eigv[2,2] = lx
    right_eigv[2,3] = u
    right_eigv[2,4] = u + c*nx

    right_eigv[3,1] = v - c*ny
    right_eigv[3,2] = ly
    right_eigv[3,3] = v
    right_eigv[3,4] = v + c*ny

    right_eigv[4,1] = H - qn*c
    right_eigv[4,2] = ql
    right_eigv[4,3] = T(0.5)*q2
    right_eigv[4,4] = H + qn*c

    return SMatrix(left_eigv), SMatrix(right_eigv)
end

function rhsmp5(e::Euler{T}, u) where T<:Real

    n = e.nx

    uL = @MVector zeros(T,nv)
    uR = @MVector zeros(T,nv)
    wL = @MVector zeros(T,nv)
    wR = @MVector zeros(T,nv)

    qi   = @MVector zeros(T,nv)
    qip1 = @MVector zeros(T,nv)
    qip2 = @MVector zeros(T,nv)
    qip3 = @MVector zeros(T,nv)
    qim1 = @MVector zeros(T,nv)
    qim2 = @MVector zeros(T,nv)

    # Loop over all cell edges except last one
    @inbounds for j in 1:n
        @inbounds for i in 3:n-3
            im1 = i - 1
            im2 = i - 2
            ip1 = i + 1
            ip2 = i + 2
            ip3 = i + 3

            uavg = T(0.5)*(e.ux[i,j]+e.ux[ip1,j])
            vavg = T(0.5)*(e.uy[i,j]+e.uy[ip1,j])
            cavg = T(0.5)*(e.c[i,j]+e.c[ip1,j])
            Havg = T(0.5)*(e.H[i,j]+e.H[ip1,j])
            lev,rev = eigenvec(1,0,uavg,vavg,cavg,Havg)

            qi   = @views lev*u[i,j,:]
            qip1 = @views lev*u[ip1,j,:]
            qip2 = @views lev*u[ip2,j,:]
            qip3 = @views lev*u[ip3,j,:]
            qim1 = @views lev*u[im1,j,:]
            qim2 = @views lev*u[im2,j,:]

            for k in 1:nv
                wL[k] = (T(2)*qim2[k] - T(13)*qim1[k] + T(47)*qi[k] + T(27)*qip1[k] - T(3)*qip2[k])/T(60)

                wp = qi[k] + minmod(qip1[k] - qi[k],T(4)*(qi[k] - qim1[k]))
                if ((wL[k]-qi[k])*(wL[k]-wp)) >= T(1e-20)
                    dm = qim2[k] - T(2)*qim1[k] + qi[k]
                    d0 = qim1[k] - T(2)*qi[k]   + qip1[k]
                    dp = qi[k]   - T(2)*qip1[k] + qip2[k]

                    dm4p = minmod4(T(4)*d0 - dp,T(4)*dp - d0, d0,dp)
                    dm4m = minmod4(T(4)*d0 - dm,T(4)*dm - d0, d0,dm)

                    uul = qi[k] + T(4)*(qi[k] - qim1[k])
                    uav = T(0.5)*(qi[k] + qip1[k])
                    umd = uav - T(0.5)*dm4p
                    ulc = qi[k] + T(0.5)*(qi[k] - qim1[k]) + T(4/3)*dm4m

                    umin = max(min(qi[k],qip1[k],umd),min(qi[k],uul,ulc))
                    umax = min(max(qi[k],qip1[k],umd),max(qi[k],uul,ulc))

                    wL[k] = wL[k] + minmod(umin - wL[k],umax - wL[k])
                end 

                wR[k] = (T(2)*qip3[k] - T(13)*qip2[k] + T(47)*qip1[k] + T(27)*qi[k] - T(3)*qim1[k])/T(60)

                wp = qip1[k] + minmod(qi[k] - qip1[k],T(4)*(qip1[k] - qip2[k]))
                if (wR[k]-qip1[k])*(wR[k]-wp) >= T(1e-20)
                    dm = qip3[k] - T(2)*qip2[k] + qip1[k]
                    d0 = qip2[k] - T(2)*qip1[k] + qi[k]
                    dp = qip1[k] - T(2)*qi[k]   + qim1[k]

                    dm4p = minmod4(T(4)*d0 - dm,T(4)*dm - d0, d0,dm)
                    dm4m = minmod4(T(4)*d0 - dp,T(4)*dp - d0, d0,dp)

                    uul = qip1[k] + T(4)*(qip1[k] - qip2[k])
                    uav = T(0.5)*(qi[k] + qip1[k])
                    umd = uav - T(0.5)*dm4m
                    ulc = qip1[k] + T(0.5)*(qip1[k] - qip2[k]) + T(4/3)*dm4p

                    umin = max(min(qip1[k],qi[k],umd),min(qip1[k],uul,ulc))
                    umax = min(max(qip1[k],qi[k],umd),max(qip1[k],uul,ulc))

                    wR[k] = wR[k] + minmod(umin - wR[k],umax - wR[k])
                end
            end
            uL = rev*wL
            uR = rev*wR
        
            e.fh[i,j,:] = riemann_solvers(T(1),T(0),SVector(uL),SVector(uR))
        end
    end
        
# Loop over all cell edges except last one
    @inbounds for i in 3:n-3
        @inbounds for j in 1:n
            im1 = i - 1
            im2 = i - 2
            ip1 = i + 1
            ip2 = i + 2
            ip3 = i + 3

            uavg = T(0.5)*(e.ux[j,i]+e.ux[j,ip1])
            vavg = T(0.5)*(e.uy[j,i]+e.uy[j,ip1])
            cavg = T(0.5)*(e.c[j,i]+e.c[j,ip1])
            Havg = T(0.5)*(e.H[j,i]+e.H[j,ip1])
            lev,rev = eigenvec(0,1,uavg,vavg,cavg,Havg)

            qi   = @views lev*u[j,i,:]
            qip1 = @views lev*u[j,ip1,:]
            qip2 = @views lev*u[j,ip2,:]
            qip3 = @views lev*u[j,ip3,:]
            qim1 = @views lev*u[j,im1,:]
            qim2 = @views lev*u[j,im2,:]

            for k in 1:nv
                wL[k] = (T(2)*qim2[k] - T(13)*qim1[k] + T(47)*qi[k] + T(27)*qip1[k] - T(3)*qip2[k])/T(60)

                wp = qi[k] + minmod(qip1[k] - qi[k],T(4)*(qi[k] - qim1[k]))
                if ((wL[k]-qi[k])*(wL[k]-wp)) >= T(1e-20)
                    dm = qim2[k] - T(2)*qim1[k] + qi[k]
                    d0 = qim1[k] - T(2)*qi[k]   + qip1[k]
                    dp = qi[k]   - T(2)*qip1[k] + qip2[k]

                    dm4p = minmod4(T(4)*d0 - dp,T(4)*dp - d0, d0,dp)
                    dm4m = minmod4(T(4)*d0 - dm,T(4)*dm - d0, d0,dm)

                    uul = qi[k] + T(4)*(qi[k] - qim1[k])
                    uav = T(0.5)*(qi[k] + qip1[k])
                    umd = uav - T(0.5)*dm4p
                    ulc = qi[k] + T(0.5)*(qi[k] - qim1[k]) + T(4/3)*dm4m

                    umin = max(min(qi[k],qip1[k],umd),min(qi[k],uul,ulc))
                    umax = min(max(qi[k],qip1[k],umd),max(qi[k],uul,ulc))

                    wL[k] = wL[k] + minmod(umin - wL[k],umax - wL[k])
                end

                wR[k] = (T(2)*qip3[k] - T(13)*qip2[k] + T(47)*qip1[k] + T(27)*qi[k] - T(3)*qim1[k])/T(60)

                wp = qip1[k] + minmod(qi[k] - qip1[k],T(4)*(qip1[k] - qip2[k]))
                if (wR[k]-qip1[k])*(wR[k]-wp) >= T(1e-20)
                    dm = qip3[k] - T(2)*qip2[k] + qip1[k]
                    d0 = qip2[k] - T(2)*qip1[k] + qi[k]
                    dp = qip1[k] - T(2)*qi[k]   + qim1[k]

                    dm4p = minmod4(T(4)*d0 - dm,T(4)*dm - d0, d0,dm)
                    dm4m = minmod4(T(4)*d0 - dp,T(4)*dp - d0, d0,dp)

                    uul = qip1[k] + T(4)*(qip1[k] - qip2[k])
                    uav = T(0.5)*(qi[k] + qip1[k])
                    umd = uav - T(0.5)*dm4m
                    ulc = qip1[k] + T(0.5)*(qip1[k] - qip2[k]) + T(4/3)*dm4p

                    umin = max(min(qip1[k],qi[k],umd),min(qip1[k],uul,ulc))
                    umax = min(max(qip1[k],qi[k],umd),max(qip1[k],uul,ulc))

                    wR[k] = wR[k] + minmod(umin - wR[k],umax - wR[k])
                end

            end

            uL = rev*wL
            uR = rev*wR

            e.gh[j,i,:] = riemann_solvers(T(0),T(1),SVector(uL),SVector(uR))
        end
    end

    # Do nothing BCs
    e.fh[n-2,:,:] .= @view e.fh[n-3,:,:]
    e.fh[n-1,:,:] .= @view e.fh[n-3,:,:]
    e.fh[n,:,:]   .= @view e.fh[n-3,:,:]
    e.fh[2,:,:]   .= @view e.fh[3,:,:]
    e.fh[1,:,:]   .= @view e.fh[3,:,:]

    e.gh[:,n-2,:] .= @view e.gh[:,n-3,:]
    e.gh[:,n-1,:] .= @view e.gh[:,n-3,:]
    e.gh[:,n,:]   .= @view e.gh[:,n-3,:]
    e.gh[:,2,:]   .= @view e.gh[:,3,:]
    e.gh[:,1,:]   .= @view e.gh[:,3,:]

    for j=2:n
        for i=2:n
            for k=1:nv
                e.b[i,j,k] = -(e.fh[i,j,k] - e.fh[i-1,j,k])/e.dx - (e.gh[i,j,k] - e.gh[i,j-1,k])/e.dy
            end
        end
    end

    e.b[1,:,:] .= @view e.b[2,:,:]
    e.b[:,1,:] .= @view e.b[:,2,:]
end

@inline function minmod(a,b)
    return 0.5*(sign(a) + sign(b))*min(abs(a),abs(b))
end

@inline function minmod4(a,b,c,d)
    return 0.125*(sign(a) + sign(b))*
            abs((sign(a) + sign(b))*
            (sign(a) + sign(d)))*
            min(abs(a),abs(b),abs(c),abs(d))
end


function rhsweno5(e::Euler{T}, u) where T<:Real
    n = e.nx

    a1 = T(13.0/12.0)
    a2 = T(0.25)

    beta = @MVector zeros(T,3)
    w = @MVector zeros(T,3)
    fht = @MVector zeros(T,3)
    uL = @MVector zeros(T,nv)
    uR = @MVector zeros(T,nv)
    wL = @MVector zeros(T,nv)
    wR = @MVector zeros(T,nv)
    fL = @MVector zeros(T,nv)
    fR = @MVector zeros(T,nv)
    sx = @MVector zeros(T,nv)
    sy = @MVector zeros(T,nv)

    sx[1] = maximum(i -> abs(e.ux[i] - e.c[i]), eachindex(e.ux))
    sx[2] = maximum(abs, e.ux)
    sx[3] = maximum(abs, e.ux)
    sx[4] = maximum(i -> abs(e.ux[i] + e.c[i]), eachindex(e.ux))

    sy[1] = maximum(i -> abs(e.uy[i] - e.c[i]), eachindex(e.uy))
    sy[2] = maximum(abs, e.uy)
    sy[3] = maximum(abs, e.uy)
    sy[4] = maximum(i -> abs(e.uy[i] + e.c[i]), eachindex(e.uy))

    qi = @MVector zeros(T,nv)
    qip1 = @MVector zeros(T,nv)
    qip2 = @MVector zeros(T,nv)
    qip3 = @MVector zeros(T,nv)
    qim1 = @MVector zeros(T,nv)
    qim2 = @MVector zeros(T,nv)


    # Loop over all cell edges except last one
    @inbounds for j in 1:n
        @inbounds for i in 3:n-3
            im1 = i - 1
            im2 = i - 2
            ip1 = i + 1
            ip2 = i + 2
            ip3 = i + 3

            uavg = T(0.5)*(e.ux[i,j]+e.ux[ip1,j])
            vavg = T(0.5)*(e.uy[i,j]+e.uy[ip1,j])
            cavg = T(0.5)*(e.c[i,j]+e.c[ip1,j])
            Havg = T(0.5)*(e.H[i,j]+e.H[ip1,j])
            lev,rev = eigenvec(1,0,uavg,vavg,cavg,Havg)

            qi   = @views lev*u[i,j,:]
            qip1 = @views lev*u[ip1,j,:]
            qip2 = @views lev*u[ip2,j,:]
            qip3 = @views lev*u[ip3,j,:]
            qim1 = @views lev*u[im1,j,:]
            qim2 = @views lev*u[im2,j,:]

            @inbounds @fastmath for k = 1:4
                beta[1] = a1*(qi[k]-T(2)*qip1[k]+qip2[k])^2 +
                      a2*(T(3)*qi[k]-T(4)*qip1[k]+qip2[k])^2
                beta[2] = a1*(qim1[k]-T(2)*qi[k]+qip1[k])^2 +
                      a2*(qim1[k]-qip1[k])^2
                beta[3] = a1*(qim2[k]-T(2)*qim1[k]+qi[k])^2 +
                      a2*(qim2[k]-T(4)*qim1[k]+T(3)*qi[k])^2

                w = wenozpi_weights(e,SVector(beta),pg)

                fht[3] = T(1/3.) *qim2[k] - T(7/6.)*qim1[k] + T(11/6.) *qi[k]
                fht[2] = T(-1/6.) *qim1[k] + T(5/6.) *qi[k] + T(1/3.) *qip1[k]
                fht[1] = T(1/3.) *qi[k] + T(5/6.) *qip1[k] - T(1/6.) *qip2[k]

                wL[k] = w[1]*fht[1] + w[2]*fht[2] + w[3]*fht[3]

                beta[1] = a1*(qip1[k]-T(2)*qip2[k]+qip3[k])^2 +
                          a2*(T(3)*qip1[k]-T(4)*qip2[k]+qip3[k])^2
                beta[2] = a1*(qi[k]-T(2)*qip1[k]+qip2[k])^2 +
                          a2*(qi[k]-qip2[k])^2
                beta[3] = a1*(qim1[k]-T(2)*qi[k]+qip1[k])^2 +
                          a2*(qim1[k]-T(4)*qi[k]+T(3)*qip1[k])^2

                w = wenozp_weights(e,SVector(beta),mg)

                fht[3] = -T(1/6.) *qim1[k] + T(5/6.) *qi[k] + T(1/3.) *qip1[k]
                fht[2] = T(1/3.) *qi[k] + T(5/6.) *qip1[k] - T(1/6.) *qip2[k]
                fht[1] = T(11/6.) *qip1[k] - T(7/6.) *qip2[k] + T(1/3.) *qip3[k]

                wR[k] = w[1]*fht[1] + w[2]*fht[2] + w[3]*fht[3]

            end

            uL = rev*wL
            uR = rev*wR

            e.fh[i,j,:] = riemann_solvers(T(1),T(0),SVector(uL),SVector(uR))

            # fL = localfluxx(SVector(uL))
            # fR = localfluxx(SVector(uR))

            # for k = 1:4
            #     e.fh[i,j,k] = 0.5*(fL[k] + fR[k] - sx[4]*(uR[k]-uL[k]))
            #end
        end

    end

    # Loop over all cell edges except last one
    @inbounds for i in 3:n-3
        @inbounds for j in 1:n
            im1 = i - 1
            im2 = i - 2
            ip1 = i + 1
            ip2 = i + 2
            ip3 = i + 3

            uavg = T(0.5)*(e.ux[j,i]+e.ux[j,ip1])
            vavg = T(0.5)*(e.uy[j,i]+e.uy[j,ip1])
            cavg = T(0.5)*(e.c[j,i]+e.c[j,ip1])
            Havg = T(0.5)*(e.H[j,i]+e.H[j,ip1])
            lev,rev = eigenvec(0,1,uavg,vavg,cavg,Havg)

            qi   = @views lev*u[j,i,:]
            qip1 = @views lev*u[j,ip1,:]
            qip2 = @views lev*u[j,ip2,:]
            qip3 = @views lev*u[j,ip3,:]
            qim1 = @views lev*u[j,im1,:]
            qim2 = @views lev*u[j,im2,:]

            @inbounds @fastmath for k = 1:4
                beta[1] = a1*(qi[k]-T(2)*qip1[k]+qip2[k])^2 +
                          a2*(T(3)*qi[k]-T(4)*qip1[k]+qip2[k])^2
                beta[2] = a1*(qim1[k]-T(2)*qi[k]+qip1[k])^2 +
                          a2*(qim1[k]-qip1[k])^2
                beta[3] = a1*(qim2[k]-T(2)*qim1[k]+qi[k])^2 +
                          a2*(qim2[k]-T(4)*qim1[k]+T(3)*qi[k])^2

                w = wenozpi_weights(e,SVector(beta),pg)

                fht[3] = T(1/3.) *qim2[k] - T(7/6.) *qim1[k] + T(11/6.) *qi[k]
                fht[2] = T(-1/6.) *qim1[k] + T(5/6.) *qi[k] + T(1/3.) *qip1[k]
                fht[1] = T(1/3.) *qi[k] + T(5/6.) *qip1[k] - T(1/6.) *qip2[k]

                wL[k] = w[1]*fht[1] + w[2]*fht[2] + w[3]*fht[3]

                beta[1] = a1*(qip1[k]-T(2)*qip2[k]+qip3[k])^2 +
                          a2*(T(3)*qip1[k]-T(4)*qip2[k]+qip3[k])^2
                beta[2] = a1*(qi[k]-T(2)*qip1[k]+qip2[k])^2 +
                          a2*(qi[k]-qip2[k])^2
                beta[3] = a1*(qim1[k]-T(2)*qi[k]+qip1[k])^2 +
                          a2*(qim1[k]-T(4)*qi[k]+T(3)*qip1[k])^2

                w = wenozp_weights(e,SVector(beta),mg)

                fht[3] = -T(1/6.) *qim1[k] + T(5/6.) *qi[k] + T(1/3.) *qip1[k]
                fht[2] = T(1/3.) *qi[k] + T(5/6.) *qip1[k] - T(1/6.) *qip2[k]
                fht[1] = T(11/6.) *qip1[k] - T(7/6.) *qip2[k] + T(1/3.) *qip3[k]

                wR[k] = w[1]*fht[1] + w[2]*fht[2] + w[3]*fht[3]

            end

            uL = rev*wL
            uR = rev*wR

            e.gh[j,i,:] = riemann_solvers(T(0),T(1),SVector(uL),SVector(uR))

            # fL = localfluxy(SVector(uL))
            # fR = localfluxy(SVector(uR))

            # for k = 1:4
            #     e.gh[j,i,k] = 0.5*(fL[k] + fR[k] - sy[4]*(uR[k]-uL[k]))
            # end
        end
    end

    # Do nothing BCs
    e.fh[n-2,:,:] .= @view e.fh[n-3,:,:]
    e.fh[n-1,:,:] .= @view e.fh[n-3,:,:]
    e.fh[n,:,:]   .= @view e.fh[n-3,:,:]
    e.fh[2,:,:]   .= @view e.fh[3,:,:]
    e.fh[1,:,:]   .= @view e.fh[3,:,:]

    e.gh[:,n-2,:] .= @view e.gh[:,n-3,:]
    e.gh[:,n-1,:] .= @view e.gh[:,n-3,:]
    e.gh[:,n,:]   .= @view e.gh[:,n-3,:]
    e.gh[:,2,:]   .= @view e.gh[:,3,:]
    e.gh[:,1,:]   .= @view e.gh[:,3,:]

    for j=2:n
        for i=2:n
            for k=1:nv
                e.b[i,j,k] = -(e.fh[i,j,k] - e.fh[i-1,j,k])/e.dx - (e.gh[i,j,k] - e.gh[i,j-1,k])/e.dy
            end
        end
    end

    e.b[1,:,:] .= @view e.b[2,:,:]
    e.b[:,1,:] .= @view e.b[:,2,:]
end

# WENO-Z+ (Acker et al. JCP, 313, 2016)
@inline function wenozp_weights(e::Euler{T}, beta, lw) where T<:Real
    λ = e.dx # sqrt(dx)
    wt = @MVector zeros(T,3)
    τ = abs(beta[1]-beta[3])
    ε = T(1e-20)
    for l in 1:3
        wt[l] = lw[l]*(T(1)+(τ/(beta[l]+ε)^2) + λ*(beta[l]+ε)/(τ+ε))
    end
    SVector(wt/sum(wt))
end

# WENO-Z+I (Luo et al. C&F, 218, 2021)
# Improvement of the WENO-Z+ Scheme
@inline function wenozpi_weights(e::Euler{T}, beta, lw) where T<:Real
    λ = e.dx^(T(2/3)) # sqrt(dx)
    ε = T(1e-20)
    wt = @MVector zeros(T,3)
    βmin = min(beta[1],beta[2],beta[3])
    βmax = max(beta[1],beta[2],beta[3])
    α = T(1) - βmin/(βmax+ε)
    τ = abs(beta[1]-beta[3])
    
    for l in 1:3
        wt[l] = lw[l]*(T(1)+(τ/(beta[l]+ε)^2) + λ*α*beta[l]/(βmax+ε))
    end
    SVector(wt/sum(wt))
end

function saxpy!(a::AbstractArray{T}, b::AbstractArray{T}, m::T, c::AbstractArray{T}) where T<:Real
    ## a = b + m*c
    n = length(a)
    @inbounds @fastmath @simd for i in 1:n
        a[i] = b[i] + m*c[i]
    end
end

function saxpbypcz!(s::AbstractArray{T}, 
                    a::T, x::AbstractArray{T},
                    b::T, y::AbstractArray{T},
                    c::T, z::AbstractArray{T}) where T<:Real
    ## s = a*x +b*y + c*z
    n = length(s)
    @inbounds @fastmath @simd for i in 1:n
        s[i] = a*x[i] + b*y[i] + c*z[i]
    end
end

function rk3tvdweno5(e::Euler{T}, u) where T<:Real
    # Stage 1
    decomp(e, u)
    dt = timestep(e)
    rhsweno5(e, u)
    #@. e.u1 = u + dt*e.b
    saxpy!(e.u1, u, dt, e.b)

    # Stage 2
    decomp(e, e.u1)
    dt = timestep(e)
    rhsweno5(e, e.u1)
    #@. e.u2 = T(0.75)*u + T(0.25)*e.u1 + T(0.25)*dt*e.b
    saxpbypcz!(e.u2, T(0.75), u, T(0.25), e.u1, T(0.25)*dt, e.b)
    # Stage 3
    decomp(e, e.u2)
    dt = timestep(e)
    rhsweno5(e, e.u2)
    #@. u = T(1.0/3.0)*u + T(2.0/3.0)*e.u2 + T(2.0/3.0)*dt*e.b
    saxpbypcz!(u, T(1/3.0), u, T(2.0/3.0), e.u2, T(2.0/3.0)*dt, e.b)
end

function rk3tvdmp5(e::Euler{T}, u) where T<:Real
    # Stage 1
    decomp(e, u)
    dt = timestep(e)
    rhsmp5(e, u)
    #@. e.u1 = u + dt*e.b
    saxpy!(e.u1, u, dt, e.b)

    # Stage 2
    decomp(e, e.u1)
    dt = timestep(e)
    rhsmp5(e, e.u1)
    #@. e.u2 = T(0.75)*u + T(0.25)*e.u1 + T(0.25)*dt*e.b
    saxpbypcz!(e.u2, T(0.75), u, T(0.25), e.u1, T(0.25)*dt, e.b)
    # Stage 3
    decomp(e, e.u2)
    dt = timestep(e)
    rhsmp5(e, e.u2)
    #@. u = T(1.0/3.0)*u + T(2.0/3.0)*e.u2 + T(2.0/3.0)*dt*e.b
    saxpbypcz!(u, T(1/3.0), u, T(2.0/3.0), e.u2, T(2.0/3.0)*dt, e.b)
end

function timestep(e::Euler{T}) where T<:Real
    dtx = e.dx/maximum(i -> abs(e.ux[i]) + e.c[i], eachindex(e.ux))
    dty = e.dy/maximum(i -> abs(e.uy[i]) + e.c[i], eachindex(e.uy))
    dt  = e.cfl*dtx*dty/(dtx+dty)
    T(dt)
end

end
