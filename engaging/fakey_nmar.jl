using Pkg
Pkg.activate("..")

using Revise
using PHD

using Random, Statistics, CSV, DataFrames, LinearAlgebra

dataset_list = PHD.list_datasets(p_min = 1)
@show length(dataset_list)

# SNR_list = [2, 1, .5]
missingsignal_list = [0,1,2,3,4,5,6,7,8,9,10]

if !isdir("../results")
    mkdir("../results")
end
savedir = "../results/fakey_nmar/"
if !isdir(savedir)
    mkdir(savedir)
end
SNR = 2

affine_on_static_only = true

results_main = DataFrame(dataset=[], SNR=[], k=[], kMissing=[], splitnum=[], method=[], osr2=[])

for ARG in ARGS
    array_num = parse(Int, ARG)
    d_num = mod(array_num, length(dataset_list)) + 1
    aux_num = div(array_num, length(dataset_list)) + 1

    dname = dataset_list[d_num]#"dermatology" #"""thyroid-disease-thyroid-0387" #dataset_list[1]
    k_missingsignal = missingsignal_list[aux_num]
    @show dname, k_missingsignal

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
    for j in PHD.unique_missing_patterns(X_missing)
        delete_obs[j] = false
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
    Random.seed!(5234)
    @time Y, k, k_missing = PHD.linear_y(X_full, X_missing, k=10, SNR=SNR, canbemissing=canbemissing, k_missing_in_signal=k_missingsignal, mar=false) ;

    if k_missing == k_missingsignal #If not enough missing features to generate Y with k_missingsignal, abort (already done)
        test_prop = .3

        for iter in 1:10
            results_table = similar(results_main,0)

            filename = string(dname, "_SNR_", SNR, "_nmiss_", k_missingsignal, "_$iter.csv")

            # Split train / test
            Random.seed!(56802+767*iter)
            test_ind = PHD.split_dataset(X_missing, test_fraction = test_prop)

            ## Method Oracle
            df = X_full[:,:]
            df[!,:Test] = test_ind
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Oracle", OSR2])
            CSV.write(savedir*filename, results_table)

            df = [X_full[:,:] PHD.indicatemissing(X_missing[:,:]; removezerocols=true)]
            df[!,:Test] = test_ind
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Oracle XM", OSR2])
            CSV.write(savedir*filename, results_table)


            ## Method 0
            try
                df = X_missing[:,.!canbemissing] #This step can raise an error if all features can be missing
                df[!,:Test] = test_ind
                linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
                R2, OSR2 = PHD.evaluate(Y, df, linear)
                push!(results_table, [dname, SNR, k, k_missing, iter, "Complete Features", OSR2])
            catch #In this case, simply predict the mean - which leads to 0. OSR2
                push!(results_table, [dname, SNR, k, k_missing, iter, "Complete Features", 0.])
            end
            CSV.write(savedir*filename, results_table)

            ## Method 1.1
            X_imputed = PHD.mice_bruteforce(X_missing);
            df = deepcopy(X_imputed)
            df[!,:Test] = test_ind
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 1", OSR2])
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
            push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 2", OSR2])
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
            push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 3", OSR2])
            CSV.write(savedir*filename, results_table)

            ## Method 1.4
            means_df = PHD.compute_mean(X_missing[.!test_ind,:])
            X_imputed = PHD.mean_impute(X_missing, means_df);
            df = deepcopy(X_imputed)
            df[!,:Test] = test_ind
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 4", OSR2])
            CSV.write(savedir*filename, results_table)

            ## Method 1.5 Mean and mode impute
            means_df = PHD.compute_mean(X_missing[.!test_ind,:])
            X_imputed = PHD.mean_impute(X_missing, means_df);
            df = deepcopy(X_imputed)
            PHD.mode_impute!(df, train = .!test_ind)
            df[!,:Test] = test_ind
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 5", OSR2])
            CSV.write(savedir*filename, results_table)

            ## Method 2: Static Adaptability
            df = deepcopy(X_missing)
            df[!,:Test] = test_ind
            X_augmented = hcat(PHD.zeroimpute(df), PHD.indicatemissing(df, removezerocols=true))
            linear2, bestparams2 = PHD.regress_cv(Y, X_augmented, lasso=[true],
                                                    alpha=collect(0.1:0.1:1),
                                                    missing_penalty=[2.0,4.0,6.0,8.0,12.0,16.0])
            R2, OSR2 = PHD.evaluate(Y, X_augmented, linear2)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Static", OSR2])
            CSV.write(savedir*filename, results_table)

            ## Method 3: Affine Adaptability
            df = deepcopy(X_missing)
            df[!,:Test] = test_ind
            sub_features = names(df)
            if affine_on_static_only
                aux = names(X_augmented)[findall(abs.(convert(Array, linear2[1,:])) .> 0)]
                sub_features = intersect(sub_features, unique(map(t -> split(t, "_missing")[1], aux)))
                push!(sub_features, "Test")
            end
            X_affine = PHD.augmentaffine(df[:,sub_features], removezerocols=true)
            linear3, bestparams3 = PHD.regress_cv(Y, X_affine, lasso=[true], alpha=collect(0.1:0.1:1),
                                                  missing_penalty=[2.0,4.0,6.0,8.0,12.0,16.0])
            R2, OSR2 = PHD.evaluate(Y, X_affine, linear3)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Affine", OSR2])
            CSV.write(savedir*filename, results_table)
        end
    end
end
