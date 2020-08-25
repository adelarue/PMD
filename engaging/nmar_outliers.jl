using Pkg
Pkg.activate("..")

using Revise
using PHD
using Random, Statistics, CSV, DataFrames, LinearAlgebra

dataset_list = PHD.list_datasets(p_min = 1)
k_list = [5, 10]
SNR = 2

# save information
if !isdir("../results")
    mkdir("../results")
end
savedir = "../results/nmar_outliers/"
if !isdir(savedir)
    mkdir(savedir)
end
results_main = DataFrame(dataset=[], SNR=[], k=[], kMissing=[], splitnum=[], method=[],
                         r2 = [], osr2=[])

id = 1
if length(ARGS) > 0
	id = ARGS[1]
end

counter = 0
for dname in dataset_list, k in k_list, k_missingsignal in 0:k
	global counter += 1
	if counter != id
		continue
	end
	@show dname, k, k_missingsignal

	# Read in a data file
	X_missing = PHD.standardize_colnames(DataFrame(CSV.read("../datasets/"*dname*"/X_missing.csv",
	                                                        missingstrings=["", "NaN"])));
	@show nrow(X_missing), ncol(X_missing)
	X_full = PHD.standardize_colnames(DataFrame(CSV.read("../datasets/"*dname*"/X_full.csv")))[:,:];

	# optimize missingness pattern for outlier suppression
	X_missing = PHD.optimize_missingness(X_missing, X_full);

	# which columns can be missing
	canbemissing = [any(ismissing.(X_missing[:,j])) for j in names(X_missing)]

	# Generate target
	Random.seed!(5234)
	@time Y, k, k_missing = PHD.linear_y(X_full, X_missing, k=k, SNR=SNR, canbemissing=canbemissing,
	                                     k_missing_in_signal=k_missingsignal, mar=true);
	@show k_missing
	if k_missing != k_missingsignal
		println("")
		break
	end
	test_prop = .3
	test_ind = rand(nrow(X_missing)) .< test_prop ;

	for iter in 1:10
        results_table = similar(results_main,0)

        filename = string(dname, "_SNR_", SNR, "_nmiss_", k_missingsignal, "_$iter.csv")

        # Split train / test
        Random.seed!(56802+767*iter)
        test_ind = rand(nrow(X_missing)) .< test_prop ;

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
        X_affine = PHD.augmentaffine(df, removezerocols=true)
        linear3, bestparams3 = PHD.regress_cv(Y, X_affine, lasso=[true], alpha=collect(0.1:0.1:1),
                                              missing_penalty=[2.0,4.0,6.0,8.0,12.0,16.0])
        R2, OSR2 = PHD.evaluate(Y, X_affine, linear3)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Affine", OSR2])
        CSV.write(savedir*filename, results_table)
    end
end
