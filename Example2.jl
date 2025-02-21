using LinearAlgebra, Plots, Distributions, LaTeXStrings, Random, Optim, TracyWidomBeta, DataFrames, CSV, Base.Threads
include("SpikeEstimation.jl")
include("AltMeth.jl")
imgFolder = "Figures"
tableFolder = "Tables"

t1 = Threads.@spawn begin
N = 5000 # Increase size
d = 0.5
M = convert(Int64,ceil(N/d))
σ = 1
δvec = 1:0.1:4 # Increase mesh and range
σvec = σ^2*ones(N)
MPquants = quantMP(N,d)
jmp = 5
tol = 2.5/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,N/4,sqrt(N))))
k = convert(Int64,floor(log(N)/2))
vecNbr = 1
lenδ = length(δvec)
Percent = zeros(Float64,lenδ,5)
Avrg = zeros(Float64,lenδ,5)
SampleNbr = 100 # Increase sample nbr
@threads for i=1:lenδ
    δ = δvec[i]
    σvec[1:3] = [6,5,δ]
    sqrtΣ = Diagonal(sqrt.(σvec))
    count = zeros(Float64,5)
    Sum = zeros(Float64,5)
    if δ≤1+sqrt(d)
        SpikeNbr = 2
    else
        SpikeNbr = 3
    end
    for j=1:SampleNbr
        # Normal
        X = randn(N,M)
        # # Rademacher
        # X = 2 .* rand(Bool, N, M) .- 1
        # # Uniform
        # UnifDist = Uniform(-sqrt(3),sqrt(3))
        # X = rand(UnifDist,N,M)
        # #Beta
        # BetaDist = Beta(0.5, 0.5)
        # X = rand(BetaDist, N, M)
        # X = 2*X .-1
        W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric
        evals = eigvals(W)
        BEMA0Out = BEMA0(evals,MPquants,d,0.2)
        count[1] = BEMA0Out==SpikeNbr ? count[1]+1 : count[1]
        Sum[1] += BEMA0Out
        BEMAOut = BEMA(evals,0.2,N,d;SampleNbr=100) # Increase sample nbr
        count[2] = BEMAOut==SpikeNbr ? count[2]+1 : count[2]
        Sum[2] += BEMAOut
        PassYaoOut = PassYao(evals,d,N;SampleNbr=100) # Increase sample nbr
        count[3] = PassYaoOut==SpikeNbr ? count[3]+1 : count[3]
        Sum[3] += PassYaoOut
        DDPAOut = DDPA(evals,N,d)
        count[4] = DDPAOut==SpikeNbr ? count[4]+1 : count[4]
        Sum[4] += DDPAOut
        TChol,L_list = CholList(W,tol,k,jmp,max_iter,vecNbr)
        Nbr,Loc = EstimSpike(TChol,N)
        count[5] = Nbr==SpikeNbr ? count[5]+1 : count[5]
        Sum[5] += Nbr
    end
    Percent[i,:] = count/SampleNbr
    Avrg[i,:] = Sum/SampleNbr
end
p = plot(δvec,Percent[:,5],linecolor=:red,linewidth=3,label="",ylabel="Probability of correct estimation",xlabel=L"\delta",legend=:topright,framestyle=:box)
p = scatter!(δvec,Percent[:,5],markersize=5,color=:red,marker=:circ,label="Lanczos")
p = plot!(δvec,Percent[:,1],linecolor=:blue,linewidth=3,label="")
p = scatter!(δvec,Percent[:,1],markersize=5,color=:blue,marker=:diamond,label="BEMA0")
p = plot!(δvec,Percent[:,2],linecolor=:green,linewidth=3,label="")
p = scatter!(δvec,Percent[:,2],markersize=5,color=:green,marker=:square,label="BEMA")
p = plot!(δvec,Percent[:,3],linecolor=:orange,linewidth=3,label="")
p = scatter!(δvec,Percent[:,3],markersize=5,color=:orange,marker=:star6,label="PassYao")
p = plot!(δvec,Percent[:,4],linecolor=:purple,linewidth=3,label="")
p = scatter!(δvec,Percent[:,4],markersize=5,color=:purple,marker=:xcross,label="DDPA")
savefig(p,joinpath(imgFolder, "Ex2Fig1.png"))
tb1 = DataFrame(A=δvec,B=Percent[:,1],C=Percent[:,2],D=Percent[:,3],E=Percent[:,4],F=Percent[:,5])
CSV.write(joinpath(tableFolder,"Ex2CountTb1.csv"),tb1)
tb2 = DataFrame(A=δvec,B=Avrg[:,1],C=Avrg[:,2],D=Avrg[:,3],E=Avrg[:,4],F=Avrg[:,5])
CSV.write(joinpath(tableFolder,"Ex2AvrgTb1.csv"),tb2)
end

t2 = Threads.@spawn begin
N = 5000 # Increase size
d = 0.5
M = convert(Int64,ceil(N/d))
σ = 1
δvec = 1:0.1:4 # Increase mesh and range
σvec = σ^2*ones(N)
MPquants = quantMP(N,d)
jmp = 5
tol = 2.5/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,N/4,sqrt(N))))
k = convert(Int64,floor(log(N)/2))
vecNbr = 1
lenδ = length(δvec)
Percent = zeros(Float64,lenδ,5)
Avrg = zeros(Float64,lenδ,5)
SampleNbr = 100 # Increase sample nbr
@threads for i=1:lenδ
    δ = δvec[i]
    σvec[1:3] = [6,5,δ]
    sqrtΣ = Diagonal(sqrt.(σvec))
    count = zeros(Float64,5)
    Sum = zeros(Float64,5)
    if δ≤1+sqrt(d)
        SpikeNbr = 2
    else
        SpikeNbr = 3
    end
    for j=1:SampleNbr
        # # Normal
        # X = randn(N,M)
        # Rademacher
        X = 2 .* rand(Bool, N, M) .- 1
        # # Uniform
        # UnifDist = Uniform(-sqrt(3),sqrt(3))
        # X = rand(UnifDist,N,M)
        # #Beta
        # BetaDist = Beta(0.5, 0.5)
        # X = rand(BetaDist, N, M)
        # X = 2*X .-1
        W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric
        evals = eigvals(W)
        BEMA0Out = BEMA0(evals,MPquants,d,0.2)
        count[1] = BEMA0Out==SpikeNbr ? count[1]+1 : count[1]
        Sum[1] += BEMA0Out
        BEMAOut = BEMA(evals,0.2,N,d;SampleNbr=100) # Increase sample nbr
        count[2] = BEMAOut==SpikeNbr ? count[2]+1 : count[2]
        Sum[2] += BEMAOut
        PassYaoOut = PassYao(evals,d,N;SampleNbr=100) # Increase sample nbr
        count[3] = PassYaoOut==SpikeNbr ? count[3]+1 : count[3]
        Sum[3] += PassYaoOut
        DDPAOut = DDPA(evals,N,d)
        count[4] = DDPAOut==SpikeNbr ? count[4]+1 : count[4]
        Sum[4] += DDPAOut
        TChol,L_list = CholList(W,tol,k,jmp,max_iter,vecNbr)
        Nbr,Loc = EstimSpike(TChol,N)
        count[5] = Nbr==SpikeNbr ? count[5]+1 : count[5]
        Sum[5] += Nbr
    end
    Percent[i,:] = count/SampleNbr
    Avrg[i,:] = Sum/SampleNbr
end
p = plot(δvec,Percent[:,5],linecolor=:red,linewidth=3,label="",ylabel="Probability of correct estimation",xlabel=L"\delta",legend=:topright,framestyle=:box)
p = scatter!(δvec,Percent[:,5],markersize=5,color=:red,marker=:circ,label="Lanczos")
p = plot!(δvec,Percent[:,1],linecolor=:blue,linewidth=3,label="")
p = scatter!(δvec,Percent[:,1],markersize=5,color=:blue,marker=:diamond,label="BEMA0")
p = plot!(δvec,Percent[:,2],linecolor=:green,linewidth=3,label="")
p = scatter!(δvec,Percent[:,2],markersize=5,color=:green,marker=:square,label="BEMA")
p = plot!(δvec,Percent[:,3],linecolor=:orange,linewidth=3,label="")
p = scatter!(δvec,Percent[:,3],markersize=5,color=:orange,marker=:star6,label="PassYao")
p = plot!(δvec,Percent[:,4],linecolor=:purple,linewidth=3,label="")
p = scatter!(δvec,Percent[:,4],markersize=5,color=:purple,marker=:xcross,label="DDPA")
savefig(p,joinpath(imgFolder, "Ex2Fig2.png"))
tb1 = DataFrame(A=δvec,B=Percent[:,1],C=Percent[:,2],D=Percent[:,3],E=Percent[:,4],F=Percent[:,5])
CSV.write(joinpath(tableFolder,"Ex2CountTb2.csv"),tb1)
tb2 = DataFrame(A=δvec,B=Avrg[:,1],C=Avrg[:,2],D=Avrg[:,3],E=Avrg[:,4],F=Avrg[:,5])
CSV.write(joinpath(tableFolder,"Ex2AvrgTb2.csv"),tb2)
end

t3 = Threads.@spawn begin
N = 5000 # Increase size
d = 0.5
M = convert(Int64,ceil(N/d))
σ = 1
δvec = 1:0.1:4 # Increase mesh and range
σvec = σ^2*ones(N)
MPquants = quantMP(N,d)
jmp = 5
tol = 2.5/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,N/4,sqrt(N))))
k = convert(Int64,floor(log(N)/2))
vecNbr = 1
lenδ = length(δvec)
Percent = zeros(Float64,lenδ,5)
Avrg = zeros(Float64,lenδ,5)
SampleNbr = 100 # Increase sample nbr
@threads for i=1:lenδ
    δ = δvec[i]
    σvec[1:3] = [6,5,δ]
    sqrtΣ = Diagonal(sqrt.(σvec))
    count = zeros(Float64,5)
    Sum = zeros(Float64,5)
    if δ≤1+sqrt(d)
        SpikeNbr = 2
    else
        SpikeNbr = 3
    end
    for j=1:SampleNbr
        # # Normal
        # X = randn(N,M)
        # # Rademacher
        # X = 2 .* rand(Bool, N, M) .- 1
        # Uniform
        UnifDist = Uniform(-sqrt(3),sqrt(3))
        X = rand(UnifDist,N,M)
        # #Beta
        # BetaDist = Beta(0.5, 0.5)
        # X = rand(BetaDist, N, M)
        # X = 2*X .-1
        W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric
        evals = eigvals(W)
        BEMA0Out = BEMA0(evals,MPquants,d,0.2)
        count[1] = BEMA0Out==SpikeNbr ? count[1]+1 : count[1]
        Sum[1] += BEMA0Out
        BEMAOut = BEMA(evals,0.2,N,d;SampleNbr=100) # Increase sample nbr
        count[2] = BEMAOut==SpikeNbr ? count[2]+1 : count[2]
        Sum[2] += BEMAOut
        PassYaoOut = PassYao(evals,d,N;SampleNbr=100) # Increase sample nbr
        count[3] = PassYaoOut==SpikeNbr ? count[3]+1 : count[3]
        Sum[3] += PassYaoOut
        DDPAOut = DDPA(evals,N,d)
        count[4] = DDPAOut==SpikeNbr ? count[4]+1 : count[4]
        Sum[4] += DDPAOut
        TChol,L_list = CholList(W,tol,k,jmp,max_iter,vecNbr)
        Nbr,Loc = EstimSpike(TChol,N)
        count[5] = Nbr==SpikeNbr ? count[5]+1 : count[5]
        Sum[5] += Nbr
    end
    Percent[i,:] = count/SampleNbr
    Avrg[i,:] = Sum/SampleNbr
end
p = plot(δvec,Percent[:,5],linecolor=:red,linewidth=3,label="",ylabel="Probability of correct estimation",xlabel=L"\delta",legend=:topright,framestyle=:box)
p = scatter!(δvec,Percent[:,5],markersize=5,color=:red,marker=:circ,label="Lanczos")
p = plot!(δvec,Percent[:,1],linecolor=:blue,linewidth=3,label="")
p = scatter!(δvec,Percent[:,1],markersize=5,color=:blue,marker=:diamond,label="BEMA0")
p = plot!(δvec,Percent[:,2],linecolor=:green,linewidth=3,label="")
p = scatter!(δvec,Percent[:,2],markersize=5,color=:green,marker=:square,label="BEMA")
p = plot!(δvec,Percent[:,3],linecolor=:orange,linewidth=3,label="")
p = scatter!(δvec,Percent[:,3],markersize=5,color=:orange,marker=:star6,label="PassYao")
p = plot!(δvec,Percent[:,4],linecolor=:purple,linewidth=3,label="")
p = scatter!(δvec,Percent[:,4],markersize=5,color=:purple,marker=:xcross,label="DDPA")
savefig(p,joinpath(imgFolder, "Ex2Fig3.png"))
tb1 = DataFrame(A=δvec,B=Percent[:,1],C=Percent[:,2],D=Percent[:,3],E=Percent[:,4],F=Percent[:,5])
CSV.write(joinpath(tableFolder,"Ex2CountTb3.csv"),tb1)
tb2 = DataFrame(A=δvec,B=Avrg[:,1],C=Avrg[:,2],D=Avrg[:,3],E=Avrg[:,4],F=Avrg[:,5])
CSV.write(joinpath(tableFolder,"Ex2AvrgTb3.csv"),tb2)
end

t4 = Threads.@spawn begin
N = 5000 # Increase size
d = 0.5
M = convert(Int64,ceil(N/d))
σ = 1
δvec = 1:0.1:4 # Increase mesh and range
σvec = σ^2*ones(N)
MPquants = quantMP(N,d)
jmp = 5
tol = 2.5/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,N/4,sqrt(N))))
k = convert(Int64,floor(log(N)/2))
vecNbr = 1
lenδ = length(δvec)
Percent = zeros(Float64,lenδ,5)
Avrg = zeros(Float64,lenδ,5)
SampleNbr = 100 # Increase sample nbr
@threads for i=1:lenδ
    δ = δvec[i]
    σvec[1:3] = [6,5,δ]
    sqrtΣ = Diagonal(sqrt.(σvec))
    count = zeros(Float64,5)
    Sum = zeros(Float64,5)
    if δ≤1+sqrt(d)
        SpikeNbr = 2
    else
        SpikeNbr = 3
    end
    for j=1:SampleNbr
        # # Normal
        # X = randn(N,M)
        # # Rademacher
        # X = 2 .* rand(Bool, N, M) .- 1
        # # Uniform
        # UnifDist = Uniform(-sqrt(3),sqrt(3))
        # X = rand(UnifDist,N,M)
        # #Beta
        BetaDist = Beta(0.5, 0.5)
        X = rand(BetaDist, N, M)
        X = 2*X .-1
        W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric
        evals = eigvals(W)
        BEMA0Out = BEMA0(evals,MPquants,d,0.2)
        count[1] = BEMA0Out==SpikeNbr ? count[1]+1 : count[1]
        Sum[1] += BEMA0Out
        BEMAOut = BEMA(evals,0.2,N,d;SampleNbr=100) # Increase sample nbr
        count[2] = BEMAOut==SpikeNbr ? count[2]+1 : count[2]
        Sum[2] += BEMAOut
        PassYaoOut = PassYao(evals,d,N;SampleNbr=100) # Increase sample nbr
        count[3] = PassYaoOut==SpikeNbr ? count[3]+1 : count[3]
        Sum[3] += PassYaoOut
        DDPAOut = DDPA(evals,N,d)
        count[4] = DDPAOut==SpikeNbr ? count[4]+1 : count[4]
        Sum[4] += DDPAOut
        TChol,L_list = CholList(W,tol,k,jmp,max_iter,vecNbr)
        Nbr,Loc = EstimSpike(TChol,N)
        count[5] = Nbr==SpikeNbr ? count[5]+1 : count[5]
        Sum[5] += Nbr
    end
    Percent[i,:] = count/SampleNbr
    Avrg[i,:] = Sum/SampleNbr
end
p = plot(δvec,Percent[:,5],linecolor=:red,linewidth=3,label="",ylabel="Probability of correct estimation",xlabel=L"\delta",legend=:topright,framestyle=:box)
p = scatter!(δvec,Percent[:,5],markersize=5,color=:red,marker=:circ,label="Lanczos")
p = plot!(δvec,Percent[:,1],linecolor=:blue,linewidth=3,label="")
p = scatter!(δvec,Percent[:,1],markersize=5,color=:blue,marker=:diamond,label="BEMA0")
p = plot!(δvec,Percent[:,2],linecolor=:green,linewidth=3,label="")
p = scatter!(δvec,Percent[:,2],markersize=5,color=:green,marker=:square,label="BEMA")
p = plot!(δvec,Percent[:,3],linecolor=:orange,linewidth=3,label="")
p = scatter!(δvec,Percent[:,3],markersize=5,color=:orange,marker=:star6,label="PassYao")
p = plot!(δvec,Percent[:,4],linecolor=:purple,linewidth=3,label="")
p = scatter!(δvec,Percent[:,4],markersize=5,color=:purple,marker=:xcross,label="DDPA")
savefig(p,joinpath(imgFolder, "Ex2Fig4.png"))
tb1 = DataFrame(A=δvec,B=Percent[:,1],C=Percent[:,2],D=Percent[:,3],E=Percent[:,4],F=Percent[:,5])
CSV.write(joinpath(tableFolder,"Ex2CountTb4.csv"),tb1)
tb2 = DataFrame(A=δvec,B=Avrg[:,1],C=Avrg[:,2],D=Avrg[:,3],E=Avrg[:,4],F=Avrg[:,5])
CSV.write(joinpath(tableFolder,"Ex2AvrgTb4.csv"),tb2)
end

fetch(t1)
fetch(t2)
fetch(t3)
fetch(t4)