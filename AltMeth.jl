function Bisection(f::Function,x::Float64,y::Float64;max_iter::Integer=500,tol::Float64=1e-12)
    fx = f(x)
    fy = f(y)
    if fx*fy>0
        return nothing
    end
    if fx ≈ 0
        return x
    elseif fy ≈ 0
        return y
    end
    iter = 0
    while(abs(x-y)>tol && iter<=max_iter)
        z = (x+y)/2
        fz = f(z)
        if fz ≈ 0
            return z
        elseif fx*fz < 0
           y = z
           fy = fz
        else
            x = z
            fx = fz
        end
        iter+=1
    end
    z = (x+y)/2
    fz = f(z)
    if abs(fz)<1
        return z
    else
        return nothing
    end
end

function LanczosTri(mat; k::Integer=size(mat,1), v=randn(size(mat,1)),opt::Integer=1)
    m, n = size(mat)
    Q = zeros(eltype(mat),n, k)
    q = v / norm(v)
    Q[:, 1] .= q
    d = zeros(eltype(mat),k)
    od = zeros(eltype(mat),k-1)
    z = similar(q)
    for i = 1:k
        z .= mat * q
        d[i] = dot(q, z)
        if opt==1
            Qview = @view Q[:, 1:i]
            z .-= Qview * (Qview' * z)
            z .-= Qview * (Qview' * z)
        elseif opt==2
            Qview = @view Q[:, 1:i]
            z .-= Qview * (Qview' * z)
        else
            z .-= d[i] * q
            if i > 1
                z .-= (od[i-1]) * (@view Q[:, i-1])
            end
        end
        if i < k
            od[i] = norm(z)
            if od[i]==0
                return SymTridiagonal(d[1:i], od[1:i-1]),@view Q[:,1:i]
            end
            q .= z / od[i]
            Q[:, i+1] .= q
        end
    end
    return SymTridiagonal(d, od),Q
end

function Mab(x,a,b)
    return (a+b)/2+(b-a)*x/2
end

function invMab(x,a,b)
    return (2*x)/(b-a)-(a+b)/(b-a)
end

function Legendre(N::Integer)
    A = SymTridiagonal(fill(0.0,N),[1/sqrt(4-i^(-2)) for i=1:N-1])
    Eig = eigen(A)
    evals = Eig.values
    evects = Eig.vectors
    nodes = evals
    weights = 2*(@view evects[1,1:N]).^2
    return nodes,weights
end
                                                                                                                                        
function weight_scaling(w,a,b)
    return ((b-a)/2)*w
end

function QuadInt(h,a,b,Legendre_nodes,Legendre_weights)
    x = Mab.(Legendre_nodes,a,b)
    return sum(h.(x).*weight_scaling(Legendre_weights,a,b))
end

function MP(x,d::Float64)
    dm = (1-sqrt(d))^2
    dp = (1+sqrt(d))^2
    if dm<x<dp
        return (1/(2*π*x*min(d,1)))*sqrt(x-dm)*sqrt(dp-x)
    else
        return 0
    end
end

function quantMP(N::Integer,d::Float64)
    quants = zeros(Float64,N+1)
    dm = (1-sqrt(d))^2
    dp = (1+sqrt(d))^2
    quants[end] = dp
    quants[1] = dm
    K = 200
    nodes,weights = Legendre(K)
    MPdens = x->MP(x,d)
    for i=1:N-1
        QuantEq = x->QuadInt(MPdens,x,dp,nodes,weights)-i/N
        quants[N-i+1] = Bisection(QuantEq,quants[1],quants[N-i+2])
    end
    return quants
end

function TWquant(β::Float64)
    F = TW(1)
    Bis_f = x->F(x)-(1-β)
    t = Bisection(Bis_f,-10.0,13.0)
    return t
end

function BEMA0(evals,MPquants,d::Float64,α::Float64,t;ThreshOut::Bool=false)
    p = length(evals)
    lb = convert(Int64,floor(α*p))
    ub = p-lb
    σ2 = sum((@view evals[lb:ub]).*(@view(MPquants[lb+1:ub+1])))/sum(((@view MPquants[lb+1:ub+1])).^2)
    if d<1
        M = convert(Int64,ceil(p/d))
    else
        M = p
    end
    thresh = σ2*((1+sqrt(d))^2+t*M^(-2/3)*d^(-1/6)*(1+sqrt(d))^(4/3))
    count = 0
    i = p
    while i>0 && evals[i]>thresh
        count+=1
        i-=1
    end
    if !ThreshOut
        return count
    else
        return count, thresh
    end
end

function loss(proposal,evals,α::Float64,d::Float64,N::Integer)
    M = convert(Int64,ceil(N/d))
    L = d<1 ? zeros(Float64,10, N) : zeros(Float64,10, M)
    D = Diagonal(rand(Gamma(proposal,1/proposal),N))
    mv_normal = MvNormal(zeros(N), D)
    p = d<1 ? N : M
    l1 = zeros(Float64,p)
    if d<1
        for i in 1:10
            x1 = rand(mv_normal,M)
            W = Symmetric(x1*x1'/M)
            l1 = eigvals(W)
            L[i,:] = l1'
        end
    else
        for i in 1:10
            x1 = rand(mv_normal,M)
            W = Symmetric(x1'*x1/M)
            l1 = eigvals(W)
            L[i,:] = l1'
        end
    end
    l1 = vec(mean(L,dims=1))
    lb = convert(Int64,floor(α*p))
    ub = p-lb
    k = lb:ub
    s1 = (l1[k] \ evals[k])[1]
    l1 .*= s1
    return sum((l1[k] .- evals[k]) .^ 2)
end

function BEMA(evals,α::Float64,N::Integer,d::Float64;SampleNbr::Integer=500,β::Float64=0.1,ThreshOut::Bool=false)
    res = optimize(θ->loss(θ,evals,α,d,N),0.1,50,Brent();iterations=20)
    θ = Optim.minimizer(res)
    M = convert(Int64,ceil(N/d))
    L = d<1 ? zeros(Float64,SampleNbr, N) : zeros(Float64,SampleNbr, M)
    D = Diagonal(rand(Gamma(θ,1/θ),N))
    mv_normal = MvNormal(zeros(N), D)
    p = d<1 ? N : M
    if d<1
        for i in 1:SampleNbr
            x1 = rand(mv_normal,M)
            W = Symmetric(x1*x1'/M)
            l1 = eigvals(W)
            L[i,:] = l1'
        end
    else
        for i in 1:SampleNbr
            x1 = rand(mv_normal,M)
            W = Symmetric(x1'*x1/M)
            l1 = eigvals(W)
            L[i,:] = l1'
        end
    end
    l1 .= vec(mean(L,dims=1))
    l2 = vec([quantile(col, β) for col in eachcol(L)])
    lb = convert(Int64,floor(α*p))
    ub = p-lb
    k = lb:ub
    s1 = (l1[k] \ evals[k])[1]
    thresh = maximum(l2.*s1)
    count = 0
    i=p
    while i>0 && evals[i]>thresh
        count+=1
        i-=1
    end
    if !ThreshOut
        return count
    else
        return count, thresh
    end
end

function PassYao(evals,d,N;SampleNbr=500)
    M = convert(Int64,ceil(N/d))
    p = length(evals)
    i=1
    SigmaTol = 1e-3
    Diff = zeros(Float64,SampleNbr)
    sqrtD = zeros(Float64,N,N)
    SpikeFlag = false
    while i≤p-2 && !SpikeFlag
        SigmaFlag = false
        old_σ2 = (1/(N-i))*(sum(evals[1:p-i]))
        new_σ2 = 0
        ρ = zeros(Float64,i)
        while !SigmaFlag
            for j=1:i
                Δ = (evals[p-j+1]+old_σ2-old_σ2*(N-i)/M)^2-4*evals[p-j+1]*old_σ2
                if Δ<=0
                    SigmaFlag = true
                end
                if !SigmaFlag
                    ρ[j] = ((evals[p-j+1]+old_σ2-old_σ2*(N-i)/M)+sqrt(Δ))/2
                end
            end
            new_σ2 = (1/(N-i))*(sum(evals[1:p-i])+sum(evals[p:-1:p-i+1]-ρ))
            if abs(new_σ2-old_σ2)<SigmaTol
                SigmaFlag = true
            end
            if !SigmaFlag
                old_σ2 = new_σ2
            end
        end
        σ2 = old_σ2
        σ = sqrt(σ2)
        sqrtD = Diagonal(σ*ones(N))
        if p==N
            for j=1:SampleNbr
                X = randn(N,M)
                W = sqrtD*X*X'*sqrtD/M |>Symmetric
                nullevals = eigvals(W)
                Diff[j] = nullevals[end]-nullevals[end-1]
            end
        else
            for j=1:SampleNbr
                X = randn(N,M)
                W = X'*sqrtD*sqrtD*X/M |>Symmetric
                nullevals = eigvals(W)
                Diff[j] = nullevals[end]-nullevals[end-1]
            end
        end
        sorted_Diff = sort(Diff, rev=true)
        check = convert(Int64,ceil(SampleNbr*0.02))
        thresh = (sorted_Diff[check]+sorted_Diff[check+1])/2
        if evals[p-i+1]-evals[p-i]>thresh || evals[p-i]-evals[p-i-1]>thresh
            i+=1
        else
            SpikeFlag=true
        end
    end
    return i-1
end

function PA(X,svals;np::Integer=19,α::Integer=100)
    N,M = size(X)
    p = min(N,M)
    svalsMat = zeros(Float64,p,np)
    Xperm = similar(X)
    S = zeros(p)
    for i=1:np
        Xperm = mapslices(shuffle, X, dims=2)
        SVDRes = svd(Xperm)
        svalsMat[:,i] = SVDRes.S
    end
    k=0
    idx = convert(Int64,floor(np*α/100))
    Hvec = sort(@view svalsMat[k+1,:])
    while k<N-2 && svals[k+1]>Hvec[idx]
        k+=1
        Hvec = sort(@view svalsMat[k+1,:])
    end
    return k
end

function DPA(svals,X,N::Integer,d::Float64;eps=0)
    k=0
    M = convert(Int64,ceil(N/d))
    D = diag(X*X'/M,0)
    B = -1/maximum(D)
    z = v->-1/v+d*sum((D/N)./((1 .+D*v)))
    dz = v->1/(v^2) - d*sum((D.^2/N)./((1 .+D*v).^2))
    vcrit = Bisection(dz,B,0.0)
    Uval = z(vcrit)
    ConvFlag = false
    while k<N && !ConvFlag
        if (svals[k+1]/sqrt(M))<(1+eps)*sqrt(Uval)
            ConvFlag = true
        else
            k+=1
        end
    end
    return k
end

function DDPAmin(svals,U,V,N::Integer,d::Float64;eps=0)
    k=0
    M = convert(Int64,ceil(N/d))
    ConvFlag = false
    X = zeros(Float64,N,M)
    D = zeros(Float64,N,N)
    while k<N && !ConvFlag
        @. X = (@view U[:,1:end-k])*Diagonal((@view svals[1:end-k]))*(@view V[:,1:end-k])'
        @. D = diag(X*X'/M,0)
        B = -1/maximum(D)
        z = v->-1/v+d*sum((D/N)./((1 .+D*v)))
        dz = v->1/(v^2) - d*sum((D.^2/N)./((1 .+D*v).^2))
        vcrit = Bisection(dz,B,0.0)
        Uval = z(vcrit)
        if (svals[k+1]/sqrt(M))<(1+eps)*sqrt(Uval)
            ConvFlag = true
        else
            k+=1
        end
    end
    return k
end

function DDPA(evals,p::Integer,d::Float64)
    k = 0
    ConvFlag = false
    while k≤p && !ConvFlag
        λ = evals[end-k]
        m = (1/(p-1))*sum(((@view evals[1:end-k-1]).-λ).^(-1.0))
        v = d*m-(1-d)/λ
        D = λ*m*v
        ℓ = 1/D
        dm = (1/(p-1))*sum(((@view evals[1:end-k-1]).-λ).^(-2.0))
        dv = d*dm+(1-d)/(λ^2)
        dD = m*v+λ*(m*dv+dm*v)
        cr2 = m/(dD*ℓ)
        cl2 = v/(dD*ℓ)
        if λ<4*ℓ^2*cr2*cl2
            k+=1
        else
            ConvFlag = true
        end
    end
    return k
end