function Lanczos(mat,tol,k,jmp,max_iter;v=randn(size(mat,1)),orth::Bool=true)
    m, n = size(mat)
    q = v / norm(v)
    Q = q
    d = eltype(mat)[]
    od = eltype(mat)[]
    dAvrg_old, odAvrg_old, dStd_old, odStd_old = zeros(Float64,4)
    idx = k
    i=1
    Convflag = false
    while i<=max_iter && !Convflag 
        z = mat * q
        push!(d,dot(q,z))
        if orth
            z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
            z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
        else
            z -= d[i] * q
            if i > 1
                z -= od[i-1] * Q[:, i-1]
            end
        end
        if i < max_iter
            push!(od,norm(z))
            if od[i]==0
                return SymTridiagonal(d[1:i], od[1:i-1])
            end
            q = z / od[i]
            Q = hcat(Q,q)
            if i==idx 
                dAvrg = sum(d[i-k+1:i])/k
                odAvrg = sum(od[i-k+1:i])/k
                dStd = sqrt(sum((d[i-k+1:i].-dAvrg).^2)/(k-1))
                odStd = sqrt(sum((od[i-k+1:i].-odAvrg).^2)/(k-1))
                if dStd<tol && odStd<tol && dStd_old<tol && odStd_old<tol && abs(dAvrg-dAvrg_old)<tol && abs(odAvrg-odAvrg_old)<tol
                    z = mat * q
                    push!(d,dot(q,z))
                    Convflag = true
                else
                    dAvrg_old, odAvrg_old, dStd_old, odStd_old = dAvrg, odAvrg, dStd, odStd
                    idx = idx+jmp
                end
            end
        end
        i+=1
    end
    return Convflag,SymTridiagonal(d,od)
end

function Cholesky(T::SymTridiagonal)
    n = size(T)[1]
    H = Tridiagonal(copy(T))
    for k = 1:n-1
        H[k+1,k+1] = H[k+1,k+1] - H[k+1,k]^2/H[k,k]
        H[k:k+1,k] = H[k:k+1,k]/sqrt(H[k,k])
        H[k,k+1] = 0.
    end
    H[n,n] = sqrt(H[n,n])
    return H
end

function CholList(mat,tol,k,jmp,max_iter,vecNbr;Modtol=tol,vecList=randn(size(mat,1),vecNbr))
    vects = []
    ModChol = []
    TChol = []
    N = size(mat,1)
    sizelist = zeros(Int64,vecNbr)
    for j=1:vecNbr
        v = randn(N)
        Convflag, T = Lanczos(mat,tol,k,jmp,max_iter,v=vecList[:,j])
        L = Cholesky(T)
        d = diag(L,0)
        od = diag(L,-1)
        idx = size(L,1)-1
        i,d_sum,od_Sum = zeros(Float64,3)
        if Convflag
            i = idx-jmp-k
            d_Sum= sum(d[idx-jmp-k+1:idx+1])
            od_Sum= sum(od[idx-jmp-k+1:idx])
            dAsymp = d_Sum/(jmp+k+1)
            odAsymp = od_Sum/(jmp+k)
            while i>0 && abs(d[i]-dAsymp)<Modtol && abs(od[i]-odAsymp)<Modtol
                d_Sum+=d[i]
                od_Sum+=od[i]
                i-=1
            end
            dAsymp = d_Sum/(idx-i+1)
            odAsymp = od_Sum/(idx-i)
        else
            i = max_iter-2
            d_Sum = d[max_iter-1]
            od_Sum = od[max_iter-1]
            while abs(d[i]-d[max_iter-1])<Modtol && abs(od[i]-od[max_iter-1])<Modtol
                d_Sum+=d[i]
                od_Sum+=od[i]
                i-=1
            end
            dAsymp = d_Sum/(max_iter-1-i)
            odAsymp = od_Sum/(max_iter-1-i)
        end
        d[i+1], d[i+2], od[i+1] = dAsymp, dAsymp, odAsymp
        ModL = Tridiagonal(od[1:i+1],d[1:i+2],0*od[1:i+1])
        sizelist[j] = size(ModL,1)
        push!(ModChol,ModL)
        push!(TChol,L)
    end
    dAsymp = 0
    odAsymp = 0
    for j=1:vecNbr
        dAsymp += ModChol[j][sizelist[j],sizelist[j]] + ModChol[j][sizelist[j]-1,sizelist[j]-1]
        odAsymp += ModChol[j][sizelist[j],sizelist[j]-1]
    end
    dAsymp/=(2*vecNbr)
    odAsymp/=vecNbr
    for j=1:vecNbr
        ModChol[j][sizelist[j],sizelist[j]] = dAsymp
        ModChol[j][sizelist[j]-1,sizelist[j]-1] = dAsymp
        ModChol[j][sizelist[j],sizelist[j]-1] = odAsymp
    end
    return TChol,ModChol
end

BiRel = (m,z,d,ℓ) -> 1/(-z+d^2-d^2*ℓ^2*(m/(1+ℓ^2*m)))
BiRef = (z,d,ℓ) -> (-z+d^2-ℓ^2+sqrt(z-(d+ℓ)^2)*sqrt(z-(d-ℓ)^2))/(2*z*ℓ^2)
function BiRec(z,L,N;eps=1e-3)
    n = size(L,1)
    eps_min = min(N^(-1/6),eps)
    m0 = BiRef(z+1im*eps_min,L[n-1,n-1],L[n,n-1])
    for j = n-2:-1:1
            m0 = BiRel(m0,z+1im*eps_min,L[j,j],L[j+1,j])
        end
    return m0
end

function EstimSupp(ModChol)
    n = size(ModChol[1],1)
    dAsymp = ModChol[1][n,n]
    odAsymp = ModChol[1][n,n-1]
    γmin = (dAsymp-odAsymp)^2
    γplus = (dAsymp+odAsymp)^2
    return γmin, γplus
end

function EstimDensity(x,ModChol,N)
    len_x = length(x)
    dens = zeros(Float64,len_x)
    len = length(ModChol)
    for i=1:len_x
        for j=1:len
            dens[i]+=imag(BiRec(x[i],ModChol[j],N))/π
        end
    end
    return  dens/len
end

function EstimSpike(TrueChol,N;δ=0.25,c=0.5,γ=0)
    len = length(TrueChol)
    sizelist = zeros(Int64,len)
    for i=1:len
        sizelist[i] = size(TrueChol[i],1)
    end
    maxval,maxidx = findmin(sizelist)
    L = TrueChol[maxidx]
    n = size(L,1)
    γplus = γ==0 ? (L[n,n]+L[n,n-1])^2 : γ
    T = SymTridiagonal(L*L')
    evals = eigvals(T)
    Loc = []
    i = maxval
    while i>1 && evals[i]>γplus+c*N^(-δ)
        push!(Loc,evals[i])
        i-=1
    end
    Nbr = length(Loc)
    return Nbr,Loc
end