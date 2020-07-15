using Pkg
Pkg.activate("..")

using Revise
using PHD

using Random, Statistics, CSV, DataFrames, LinearAlgebra

dataset_list = [d for d in split.(read(`ls ../datasets/`, String), "\n") if length(d) > 0]
sort!(dataset_list)

if !isdir("../results")
    mkdir("../results")
end
savedir = "../results/realy/"
if !isdir(savedir)
    mkdir(savedir)
end
results_main = DataFrame(dataset=[], splitnum=[], method=[], osr2=[])

for ARG in ARGS
    array_num = parse(Int, ARG)
    d_num = mod(array_num, 71) + 1
    iter = div(array_num,71) + 1

    dname = dataset_list[d_num]#"dermatology" #"""thyroid-disease-thyroid-0387" #dataset_list[1]

    @show dname
    pb_list =  ["MASS-Cars93", "rpart-car.test.frame", "soybean-large", "thyroid-disease-thyroid-0387"]
    if dname ∈ pb_list
        # Read in a data file.
        X_missing = PHD.standardize_colnames(DataFrame(CSV.read("../datasets/"*dname*"/X_missing.csv", missingstrings=["", "NaN"]))) #df with missing values

        # Clean up : to be checked, some datasets have strings in features
        delete_obs = trues(Base.size(X_missing,1))
        for j in names(X_missing)
            if Symbol(j) != :Id && (eltype(X_missing[:,j]) == String || eltype(X_missing[:,j]) == Union{Missing,String})
                newcol = tryparse.(Float64, X_missing[:,j])
                delete_obs[newcol .== nothing] .= false
                newcol = convert(Array{Union{Float64,Missing,Nothing}}, newcol)
                newcol[newcol .== nothing] .= missing
                newcol = convert(Array{Union{Float64,Missing}}, newcol)
                X_missing[!,j] = newcol
            end
        end
        X_missing = X_missing[delete_obs,:];

        #Remove intrinsic indicators
        keep_cols = names(X_missing)
        for l in values(PHD.intrinsic_indicators(X_missing, correlation_threshold=0.9))
            setdiff!(keep_cols, l)
        end
        select!(X_missing, keep_cols)
        canbemissing = [any(ismissing.(X_missing[:,j])) for j in names(X_missing)] #indicator of missing features
        X_full = PHD.standardize_colnames(DataFrame(CSV.read("../datasets/"*dname*"/X_full.csv")))[delete_obs,keep_cols] #ground truth df

        # Create output
        # @time Y, k, k_missing = PHD.linear_y(X_full, soft_threshold=0.1, SNR=SNR, canbemissing=canbemissing, n_missing_in_signal=n_missingsignal) ;
        test_prop = .3

        target_list = names(DataFrame(CSV.read("../datasets/"*dname*"/Y.csv", missingstrings=["", "NaN"])))
        # @show length(target_list)

        Y = zeros(Base.size(X_missing,1))
        if length(target_list) <= 2
            target_name = setdiff(target_list, [:Id])[1]
            Y = DataFrame(CSV.read("../datasets/"*dname*"/Y.csv", missingstrings=["", "NaN"]))[delete_obs,target_name]
        else
            setdiff!(target_list, [:Id, :target])
            sort!(target_list)
            Y = DataFrame(CSV.read("../datasets/"*dname*"/Y.csv", missingstrings=["", "NaN"]))[delete_obs,target_list[1]]
        end

        ind_availtarget = .!ismissing.(Y)
        Y = 1.0 .* Y[ind_availtarget] #Remove missing entries before converting to Float64 !
        # Y = convert(Array{Float64}, 1.0 .* Y[ind_availtarget])
        X_missing = X_missing[ind_availtarget,:]
        X_full = X_full[ind_availtarget,:]

    # for iter in 1:10
        results_table = similar(results_main,0)

        filename = string(dname, "_real_Y", "_$iter.csv")

        Random.seed!(2142+767*iter)
        # Split train / test
        test_ind = rand(nrow(X_missing)) .< test_prop ;

        ## Method 0
        try
            df = X_missing[:,.!canbemissing] #This step can raise an error if all features can be missing
            df[!,:Test] = test_ind
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, iter, "Complete Features", OSR2])
        catch #In this case, simply predict the mean - which leads to 0. OSR2
            push!(results_table, [dname, iter, "Complete Features", 0.])
        end
        CSV.write(savedir*filename, results_table)

        ## Method 1.1
        X_imputed = PHD.mice_bruteforce(X_missing);
        df = deepcopy(X_imputed)
        df[!,:Test] = test_ind
        linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, iter, "Imp-then-Reg 1", OSR2])
        CSV.write(savedir*filename, results_table)

        ## Method 1.2
        df = deepcopy(X_missing)
        X_train_imputed = PHD.mice_bruteforce(df[.!test_ind,:]);
        select!(df, names(X_train_imputed))
        df[.!test_ind,:] .= X_train_imputed
        X_all_imputed = PHD.mice(df);
        df = deepcopy(X_all_imputed)
        df[!,:Test] = test_ind
        linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, iter, "Imp-then-Reg 2", OSR2])
        CSV.write(savedir*filename, results_table)

        ## Method 1.3
        df = deepcopy(X_missing)
        X_train_imputed = PHD.mice_bruteforce(df[.!test_ind,:]);
        X_all_imputed = PHD.mice_bruteforce(df[:,names(X_train_imputed)]);
        select!(df, names(X_train_imputed))
        df[.!test_ind,:] .= X_train_imputed
        df[test_ind,:] .= X_all_imputed[test_ind,:]
        df[!,:Test] = test_ind
        linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, iter, "Imp-then-Reg 3", OSR2])
        CSV.write(savedir*filename, results_table)

        ## Method 1.4
        means_df = PHD.compute_mean(X_missing[.!test_ind,:])
        X_imputed = PHD.mean_impute(X_missing, means_df);
        df = deepcopy(X_imputed)
        df[!,:Test] = test_ind
        linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, iter, "Imp-then-Reg 4", OSR2])
        CSV.write(savedir*filename, results_table)

        ## Method 1.5 Mean and mode impute
        means_df = PHD.compute_mean(X_missing[.!test_ind,:])
        X_imputed = PHD.mean_impute(X_missing, means_df);
        df = deepcopy(X_imputed)
        PHD.mode_impute!(df, train = .!test_ind)
        df[!,:Test] = test_ind
        linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, iter, "Imp-then-Reg 5", OSR2])
        CSV.write(savedir*filename, results_table)

        ## Method 2: Static Adaptability
        df = deepcopy(X_missing)
        df[!,:Test] = test_ind
        X_augmented = hcat(PHD.zeroimpute(df), PHD.indicatemissing(df, removezerocols=true))
        linear2, bestparams2 = PHD.regress_cv(Y, X_augmented, lasso=[true],
                                                alpha=collect(0.1:0.1:1),
                                                missing_penalty=[2.0,4.0,6.0,8.0,12.0,16.0])
        R2, OSR2 = PHD.evaluate(Y, X_augmented, linear2)
        push!(results_table, [dname, iter, "Static", OSR2])
        CSV.write(savedir*filename, results_table)

        ## Method 3: Affine Adaptability
        df = deepcopy(X_missing)
        df[!,:Test] = test_ind
        X_affine = PHD.augmentaffine(df, removezerocols=true)
        linear3, bestparams3 = PHD.regress_cv(Y, X_affine, lasso=[true], alpha=collect(0.1:0.1:1),
                                              missing_penalty=[2.0,4.0,6.0,8.0,12.0,16.0])
        R2, OSR2 = PHD.evaluate(Y, X_affine, linear3)
        push!(results_table, [dname, iter, "Affine", OSR2])
        CSV.write(savedir*filename, results_table)
    end
end
