using Pkg
Pkg.activate("..")

using Revise
using PHD

using Random, Statistics, CSV, DataFrames, LinearAlgebra

dataset_list = [d for d in split.(read(`ls ../datasets/`, String), "\n") if length(d) > 0]
# SNR_list = [2, 1, .5]
missingsignal_list = [0,1,2,3,4,5,6,7,8,9,10]

if !isdir("../results")
    mkdir("../results")
end

SNR = 2

results_table = DataFrame(dataset=[], SNR=[], k=[], kMissing=[], splitnum=[], method=[], osr2=[])

for ARG in ARGS
    array_num = parse(Int, ARG)
    d_num = mod(array_num, 39) + 1
    aux_num = div(array_num,39) + 1

    dname = dataset_list[d_num]#"dermatology" #"""thyroid-disease-thyroid-0387" #dataset_list[1]
    n_missingsignal = missingsignal_list[aux_num]

    # Read in a data file.
    X_missing = PHD.standardize_colnames(DataFrame(CSV.read("../datasets/"*dname*"/X_missing.csv", missingstrings=["", "NaN"]))) #df with missing values
    canbemissing = [any(ismissing.(X_missing[:,j])) for j in names(X_missing)] #indicator of missing features
    X_full = PHD.standardize_colnames(DataFrame(CSV.read("../datasets/"*dname*"/X_full.csv"))) #ground truth df

    # Create output
    Random.seed!(549)
    @time Y, k, k_missing = PHD.linear_y(X_full, soft_threshold=0.1, SNR=SNR, canbemissing=canbemissing, n_missing_in_signal=n_missingsignal) ;

    test_prop = .3

    for iter in 1:10
        filename = string(dname, "_SNR_", SNR, "_nmiss_", n_missingsignal, "_$iter.csv")

        # Split train / test
        test_ind = rand(nrow(X_missing)) .< test_prop ;

        ## Method 1.1
        X_imputed = PHD.mice_bruteforce(X_missing);
        df = deepcopy(X_imputed)
        df[!,:Test] = test_ind
        linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=[0.7,0.8,0.9,1.0])
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 1", OSR2])
        CSV.write("../results/"*filename, results_table)

        ## Method 1.2
        df = deepcopy(X_missing)
        X_train_imputed = PHD.mice_bruteforce(df[.!test_ind,:]);
        select!(df, names(X_train_imputed))
        df[.!test_ind,:] .= X_train_imputed
        X_all_imputed = PHD.mice(df);
        df = deepcopy(X_all_imputed)
        df[!,:Test] = test_ind
        linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=[0.7,0.8,0.9,1.0])
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 2", OSR2])
        CSV.write("../results/"*filename, results_table)

        ## Method 1.3
        df = deepcopy(X_missing)
        X_train_imputed = PHD.mice_bruteforce(df[.!test_ind,:]);
        X_all_imputed = PHD.mice_bruteforce(df[:,names(X_train_imputed)]);
        select!(df, names(X_train_imputed))
        df[.!test_ind,:] .= X_train_imputed
        df[test_ind,:] .= X_all_imputed[test_ind,:]
        df[!,:Test] = test_ind
        linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=[0.7,0.8,0.9,1.0])
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 3", OSR2])
        CSV.write("../results/"*filename, results_table)

        ## Method 1.4
        means_df = PHD.compute_mean(X_missing[.!test_ind,:])
        X_imputed = PHD.mean_impute(X_missing, means_df);
        df = deepcopy(X_imputed)
        df[!,:Test] = test_ind
        linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=[0.7,0.8,0.9,1.0])
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 4", OSR2])
        CSV.write("../results/"*filename, results_table)

        ## Method 2: Static Adaptability
        df = deepcopy(X_missing)
        df[!,:Test] = test_ind
        X_augmented = hcat(PHD.zeroimpute(df), PHD.indicatemissing(df, removezerocols=true))
        linear2, bestparams2 = PHD.regress_cv(Y, X_augmented, lasso=[true],
                                                alpha=[0.7,0.8,0.9,1.0],
                                                missing_penalty=[2.0, 4.0, 8.0, 16.0])
        R2, OSR2 = PHD.evaluate(Y, X_augmented, linear2)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Static", OSR2])
        CSV.write("../results/"*filename, results_table)

        ## Method 3: Affine Adaptability
        df = deepcopy(X_missing)
        df[!,:Test] = test_ind
        X_affine = PHD.augmentaffine(df, removezerocols=true)
        linear3, bestparams3 = PHD.regress_cv(Y, X_affine, lasso=[true], alpha=[0.8],
                                              missing_penalty=[10.0, 20.0, 40.0, 80.0, 160.0])
        R2, OSR2 = PHD.evaluate(Y, X_affine, linear3)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Affine", OSR2])
        CSV.write("../results/"*filename, results_table)
    end
end
