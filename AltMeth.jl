function Bisection(f::Function,x::Float64,y::Float64;max_iter::Integer=1500,tol::Float64=1e-16)
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
    @assert m == n "Input matrix must be square"
    Q = zeros(eltype(mat),n, k)
    q = v / norm(v)
    Q[:, 1] = q
    d = zeros(eltype(mat),k)
    od = zeros(eltype(mat),k-1)
    for i = 1:k
        z = mat * q
        d[i] = dot(q, z)
        if opt==1
            z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
            z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
        elseif opt==2
            z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
        else
            z -= d[i] * q
            if i > 1
                z -= od[i-1] * Q[:, i-1]
            end
        end
        if i < k
            od[i] = norm(z)
            if od[i]==0
                T = SymTridiagonal(d[1:i], od[1:i-1])
                return T,Q[:,1:i]
            end
            q = z / od[i]
            Q[:, i+1] = q
        end
    end
    T = SymTridiagonal(d, od)
    return T,Q
end

function Mab(x,a,b)
    return (a+b)/2+(b-a)*x/2
end

function invMab(x,a,b)
    return (2*x)/(b-a)-(a+b)/(b-a)
end

function Legendre(N)
    A = SymTridiagonal(fill(0.0,N),[1/sqrt(4-i^(-2)) for i=1:N-1])
    Eig = eigen(A)
    evals = Eig.values
    evects = Eig.vectors
    nodes = evals
    weights = 2*evects[1,1:N].^2
    return nodes,weights
end
                                                                                                                                        
function weight_scaling(w,a,b)
    return ((b-a)/2)*w
end

function QuadInt(h,a,b,Legendre_nodes,Legendre_weights)
    x = Mab.(Legendre_nodes,a,b)
    w = h.(x).*weight_scaling(Legendre_weights,a,b)
    return sum(w)
end

function MP(x,d)
    dm = (1-sqrt(d))^2
    dp = (1+sqrt(d))^2
    if dm<x<dp
        return (1/(2*π*x*d))*sqrt(x-dm)*sqrt(dp-x)
    else
        return 0
    end
end

function quantMP(N,d)
    quants = zeros(Float64,N+1)
    dm = (1-sqrt(d))^2
    dp = (1+sqrt(d))^2
    quants[end] = dp
    quants[1] = dm
    K = 400
    nodes,weights = Legendre(K)
    MPdens = x->MP(x,d)
    for i=1:N-1
        QuantEq = x->QuadInt(MPdens,x,dp,nodes,weights)-i/N
        quants[N-i+1] = Bisection(QuantEq,quants[1],quants[N-i+2])
    end
    return quants
end

function BEMA0(evals,MPquants,d,α;β=0.1)
    N = length(evals)
    lb = convert(Int64,floor(α*N))
    ub = N-lb
    σ2 = sum(evals[lb:ub].*MPquants[lb+1:ub+1])/sum((MPquants[lb+1:ub+1]).^2)
    F = TW(1)
    Bis_f = x->F(x)-(1-β)
    t = Bisection(Bis_f,-10.0,13.0)
    M = convert(Int64,ceil(N/d))
    thresh = σ2*((1+sqrt(d))^2+t*M^(-2/3)*d^(-1/6)*(1+sqrt(d))^(4/3))
    count = 0
    i = N
    while i>0 && evals[i]>thresh
        count+=1
        i-=1
    end
    return count
end

function loss(proposal,evals,α,d,N)
    M = convert(Int64,ceil(N/d))
    L = zeros(10, N)
    for i in 1:10
        x1 = rand(MvNormal(zeros(N),Diagonal(rand(Gamma(proposal,1/proposal),N))),M)
        l1 = eigvals(Symmetric(x1*x1'/M))
        L[i,:] = l1'
    end
    l1 = vec(mean(L,dims=1))
    lb = convert(Int64,floor(α*N))
    ub = N-lb
    k = lb:ub
    s1 = (l1[k] \ evals[k])[1]
    l1 .*= s1
    return sum((l1[k] .- evals[k]) .^ 2)
end

function BEMA(evals,α,N,d;SampleNbr=500,β=0.1)
    res = optimize(θ->loss(θ,evals,α,d,N),0.1,50,Brent())
    θ = Optim.minimizer(res)
    L = zeros(SampleNbr, N)
    M = convert(Int64,ceil(N/d))
    for i in 1:SampleNbr
        x1 = rand(MvNormal(zeros(N),Diagonal(rand(Gamma(θ,1/θ),N))),M)
        l1 = eigvals(Symmetric(x1*x1'/M))
        L[i,:] = l1'
    end
    l1 = vec(mean(L,dims=1))
    l2 = vec([quantile(col, β) for col in eachcol(L)])
    lb = convert(Int64,floor(α*N))
    ub = N-lb
    k = lb:ub
    s1 = (l1[k] \ evals[k])[1]
    thresh = maximum(l2.*s1)
    count = 0
    i=N
    while i>0 && evals[i]>thresh
        count+=1
        i-=1
    end
    return count
end

function PassYao(evals,d,N;SampleNbr=500)
    M = convert(Int64,ceil(N/d))
    i=1
    SigmaTol = 1e-8
    SpikeFlag = false
    while i≤N-2 && !SpikeFlag
        SigmaFlag = false
        old_σ2 = (1/(N-i))*(sum(evals[1:N-i]))
        new_σ2 = 0
        ρ = zeros(Float64,i)
        while !SigmaFlag
            for j=1:i
                Δ = (evals[N-j+1]+old_σ2-old_σ2*(N-i)/M)^2-4*evals[N-j+1]*old_σ2
                if Δ<=0
                    SigmaFlag = true
                end
                if !SigmaFlag
                    ρ[j] = ((evals[N-j+1]+old_σ2-old_σ2*(N-i)/M)+sqrt(Δ))/2
                end
            end
            new_σ2 = (1/(N-i))*(sum(evals[1:N-i])+sum(evals[N:-1:N-i+1]-ρ))
            if abs(new_σ2-old_σ2)<SigmaTol
                SigmaFlag = true
            end
            if !SigmaFlag
                old_σ2 = new_σ2
            end
        end
        σ2 = old_σ2
        σ = sqrt(σ2)
        Diff = zeros(Float64,SampleNbr)
        sqrtD = Diagonal(σ*ones(N))
        for j=1:SampleNbr
            X = randn(N,M)
            W = sqrtD*X*X'*sqrtD/M |>Symmetric
            nullevals = eigvals(W)
            Diff[j] = nullevals[end]-nullevals[end-1]
        end
        sorted_Diff = sort(Diff, rev=true)
        thresh = (sorted_Diff[10]+sorted_Diff[11])/2
        if evals[N-i+1]-evals[N-i]>thresh || evals[N-i]-evals[N-i-1]>thresh
            i+=1
        else
            SpikeFlag=true
        end
    end
    return i-1
end

function DDPA(evals,N,d)
    M = convert(Int64,ceil(N/d))
    k = 0
    ConvFlag = false
    while k≤N && !ConvFlag
        λ = evals[end-k]
        m = (1/(N-1))*sum((evals[1:end-k-1].-λ).^(-1.0))
        v = d*m-(1-d)/λ
        D = λ*m*v
        ℓ = 1/D
        dm = (1/(N-1))*sum((evals[1:end-k-1].-λ).^(-2.0))
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