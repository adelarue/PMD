using Random, Statistics, CSV, DataFrames, LinearAlgebra

include("../impute.jl")
include("../regress.jl")
include("../augment.jl")

dataset_list = [d for d in split.(read(`ls ../datasets/`, String), "\n") if length(d) > 0]

if !isdir("../results")
    mkdir("../results")
end
if !isdir("../results/penalty")
    mkdir("../results/penalty")
end

PENALTIES = [1.0, 2.0, 4.0, 8.0, 16.0]

function softthresholding(x; λ=0.1)
    if x > λ
        return x - λ
    elseif x < -λ
        return x + λ
    else
        return 0
    end
end

SNR = 4

for dname in dataset_list[1:end]
    @show dname
    for i in 1:20
        @show i
        results_table = DataFrame(dataset=[], copynum=[], penalty=[], iter=[], method=[], osr2=[])

        X_missing = DataFrame(CSV.read("../datasets/"*dname*"/$i/X_missing.csv"))
        X_full = DataFrame(CSV.read("../datasets/"*dname*"/$i/X_full.csv"))

        cols = [d for d in names(X_full) if !any(ismissing.(X_full[:,d]))]
        X_full = X_full[:,cols]
        X_missing = X_missing[:,cols]
        for penalty in PENALTIES
            @show penalty
            for iter = 1:5
                try
                @show iter
                n,p = size(X_full)
                wtrue = softthresholding.(randn(p))
                btrue = rand(1)

                test_index = findfirst(names(X_full) .== :Test)
                wtrue[test_index] = 0.

                μ = mean(Matrix{Float64}(X_full), dims=1)
                σ = std(Matrix{Float64}(X_full), dims=1)
                σ[findall(σ .== 0)].= 1
                X_normalize = (X_full .- μ) ./ σ

                Y = Matrix{Float64}(X_normalize)*wtrue .+ btrue

                noise = randn(size(X_full,1))
                noise .*= norm(Y)/norm(noise)/SNR
                Y .+= noise

                #Method 1: Impute then regress
                X_imputed = mice(X_missing);

                linear = regress(Y, X_imputed, lasso=true, alpha=0.8)

                R2, OSR2 = evaluate(Y, X_imputed, linear)

                push!(results_table, [dname, i, penalty, iter, "Impute then regress", OSR2])

                #Method 2: Augment with missing indicator
                X_augmented = hcat(zeroimpute(X_missing), indicatemissing(X_missing))
                linear2 = regress(Y, X_augmented, lasso=true, alpha=0.8, missing_penalty=penalty)

                R2, OSR2 = evaluate(Y, X_augmented, linear2)

                push!(results_table, [dname, i, penalty, iter, "Augmented", OSR2])

                #Method 3: Affine Adjustables
                X_affine = augmentaffine(X_missing)
                linear3 = regress(Y, X_affine, lasso=true, alpha=0.8, missing_penalty=penalty)

                R2, OSR2 = evaluate(Y, X_affine, linear3)

                push!(results_table, [dname, i, penalty, iter, "Augmented Affine", OSR2])

                CSV.write("../results/penalty/"*dname*"_$i.csv", results_table)
                catch
                    ()
                end
            end
        end
    end
end
