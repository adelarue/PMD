using Pkg
Pkg.activate("..")

using PHD
using Random, Statistics, CSV, DataFrames, LinearAlgebra

# dataset_list = PHD.list_datasets(p_min = 1)
dataset_list = [d for d in split.(read(`ls ../datasets/`, String), "\n") if length(d) > 0]
sort!(dataset_list)

k = 10
SNR = 2

savedir = "../results/nmar_outliers/revisions/"
mkpath(savedir)

results_main = DataFrame(dataset=[], SNR=[], k=[], kMissing=[], splitnum=[], method=[],
                         r2 = [], osr2=[], time=[])

do_benchmark = true
do_impthenreg = true
do_static = true
do_affine = true
affine_on_static_only = true
do_finite = true

id = 1
if length(ARGS) > 0
	id = parse(Int, ARGS[1])
end

counter = 0
# for dname in dataset_list, k in k_list, k_missingsignal in 0:k
# 	global counter += 1
# 	if counter != id
# 		continue
# 	end
array_num = id
# d_num = mod(array_num, length(dataset_list)) + 1
# aux_num = div(array_num, length(dataset_list)) + 1

d_num = array_num + 1
for aux_num in 1:11

dname = dataset_list[d_num]#"dermatology" #"""thyroid-disease-thyroid-0387" #dataset_list[1]
k_missingsignal = collect(0:k)[aux_num]
@show dname, k_missingsignal


# @show dname, k, k_missingsignal

# Read in a data file
X_missing = PHD.standardize_colnames(CSV.read("../datasets/"*dname*"/X_missing.csv", DataFrame,
                                                        missingstrings=["", "NaN"]));

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
X_missing = X_missing[delete_obs, :];

#Remove intrinsic indicators
keep_cols = names(X_missing)
for l in values(PHD.intrinsic_indicators(X_missing, correlation_threshold=0.9))
    setdiff!(keep_cols, l)
end
select!(X_missing, keep_cols)

X_full = PHD.standardize_colnames(CSV.read("../datasets/"*dname*"/X_full.csv", DataFrame))[:,:];
X_full = X_full[delete_obs, keep_cols];
@show nrow(X_missing), ncol(X_missing)
@show nrow(X_full), ncol(X_full)

# optimize missingness pattern for outlier suppression
X_missing = PHD.optimize_missingness(X_missing, X_full);

# which columns can be missing
canbemissing = [any(ismissing.(X_missing[:,j])) for j in names(X_missing)]

# Generate target
Random.seed!(5234)
@time Y, k, k_missing = PHD.linear_y(X_full, X_missing, k=k, SNR=SNR, canbemissing=canbemissing,
                                     k_missing_in_signal=k_missingsignal, mar=true);
@show k_missingsignal, k_missing
if k_missing == k_missingsignal
# 	println("")
# 	break
# end
	test_prop = .3

    savedfiles = filter(t -> startswith(t, dname), readdir(savedir))
    map!(t -> replace(t, ".csv" => ""), savedfiles, savedfiles)
    filter!(t -> parse(Int, split(t, "_")[end-1]) == k_missingsignal, savedfiles)
    map!(t -> split(t, "_")[end], savedfiles, savedfiles)    
    @show savedfiles
    for iter in setdiff(1:10, parse.(Int, savedfiles))    
    # for iter in 1:10
		@show iter

        results_table = similar(results_main,0)
        filename = string(dname, "_k_", k, "_kmiss_", k_missingsignal, "_iter_$iter.csv")

        # Split train / test
        Random.seed!(56802+767*iter)
        test_ind = PHD.split_dataset(X_missing, test_fraction = test_prop, random=true)
        # @show sum(test_ind) / length(test_ind)

        if do_benchmark
			println("Benchmark methods...")
			println("####################")
            ## Method Oracle
            println("Oracle")
            df = X_full[:,:]
            df[!,:Test] = test_ind
            start = time()
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            δt = (time() - start)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Oracle", R2, OSR2, δt])
            CSV.write(savedir*filename, results_table)

            df = [X_full[:,:] PHD.indicatemissing(X_missing[:,:]; removecols=:Constant)]
            df[!,:Test] = test_ind
            start = time()
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            δt = (time() - start)
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Oracle XM", R2, OSR2, δt])
            CSV.write(savedir*filename, results_table)

            ## Method 0
            println("Method 0")
            try
                df = X_missing[:,.!canbemissing] #This step can raise an error if all features can be missing
                df[!,:Test] = test_ind
                start = time()
                linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
                δt = (time() - start)
                R2, OSR2 = PHD.evaluate(Y, df, linear)
                push!(results_table, [dname, SNR, k, k_missing, iter, "Complete Features", R2, OSR2, δt])
            catch #In this case, simply predict the mean - which leads to 0. OSR2
                push!(results_table, [dname, SNR, k, k_missing, iter, "Complete Features", 0., 0., 0.])
            end
            CSV.write(savedir*filename, results_table)
        end

        if do_impthenreg
			println("Impute-then-regress methods...")
			println("###############################")
            ## Method 1.1
            println("Method 1.1")
            start = time()
            X_imputed = PHD.mice_bruteforce(X_missing);
            δt = (time() - start)
            df = deepcopy(X_imputed)
            df[!,:Test] = test_ind
            start = time()
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            δt += (time() - start)
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 1", R2, OSR2, δt])
            CSV.write(savedir*filename, results_table)

            ## Method 1.2
            println("Method 1.2")
            df = deepcopy(X_missing)
            start = time()
            X_train_imputed = PHD.mice_bruteforce(df[.!test_ind,:]);
            δt = (time() - start)
            select!(df, names(X_train_imputed))
            df[.!test_ind,:] .= X_train_imputed
            start = time()
            X_all_imputed = PHD.mice_bruteforce(df);
            δt += (time() - start)
            df = deepcopy(X_all_imputed)
            df[!,:Test] = test_ind
            start = time()
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            δt += (time() - start)
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 2", R2, OSR2, δt])
            CSV.write(savedir*filename, results_table)

            ## Method 1.3
            println("Method 1.3")
            df = deepcopy(X_missing)
            start = time()
            X_train_imputed = PHD.mice_bruteforce(df[.!test_ind,:]);
            X_all_imputed = PHD.mice_bruteforce(df[:,names(X_train_imputed)]);
            δt = (time() - start)
            select!(df, names(X_train_imputed))
            df[.!test_ind,:] .= X_train_imputed
            df[test_ind,:] .= X_all_imputed[test_ind,:]
            df[!,:Test] = test_ind
            start = time()
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            δt += (time() - start)
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 3", R2, OSR2, δt])
            CSV.write(savedir*filename, results_table)

            ## Method 1.4
            println("Method 1.4")
            start = time()
            means_df = PHD.compute_mean(X_missing[.!test_ind,:])
            X_imputed = PHD.mean_impute(X_missing, means_df);
            δt = (time() - start)
            df = deepcopy(X_imputed)
            df[!,:Test] = test_ind
            start = time()
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            δt += (time() - start)
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 4", R2, OSR2, δt])
            CSV.write(savedir*filename, results_table)

            ## Method 1.5 Mean and mode impute
            println("Method 1.5")
            start = time()
            means_df = PHD.compute_mean(X_missing[.!test_ind,:])
            X_imputed = PHD.mean_impute(X_missing, means_df);
            δt = (time() - start)
            df = deepcopy(X_imputed)
            start = time()
            PHD.mode_impute!(df, train = .!test_ind)
            δt += (time() - start)
            df[!,:Test] = test_ind
            start = time()
            linear, bestparams = PHD.regress_cv(Y, df, lasso=[true], alpha=collect(0.1:0.1:1))
            δt += (time() - start)
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 5", R2, OSR2, δt])
            CSV.write(savedir*filename, results_table)
        end

        regtype = :missing_weight
        if do_static || do_affine
            println("Adaptive methods...")
            println("###################")
            ## Method 2: Static Adaptability
            println("Method Static")
            df = deepcopy(X_missing)
            df[!,:Test] = test_ind
            start = time()
            X_augmented = hcat(PHD.zeroimpute(df), PHD.indicatemissing(df, removecols=:Zero))
            # X_augmented = PHD.zeroimpute(df)
            linear2, bestparams2 = PHD.regress_cv(Y, X_augmented, regtype=[regtype],
                                    alpha=collect(0:0.1:1),
                                    missing_penalty=[1.0,2.0,4.0,6.0,8.0,12.0])
            δt = (time() - start)
            R2, OSR2 = PHD.evaluate(Y, X_augmented, linear2)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Static", R2, OSR2, δt])
            CSV.write(savedir*filename, results_table)

            if do_affine
                ## Method 3: Affine Adaptability
                println("Method Affine")
                df = deepcopy(X_missing)
                df[!,:Test] = test_ind

                model = names(df)
                if affine_on_static_only
                    model2 = names(linear2)[findall(abs.([linear2[1,c] for c in names(linear2)]) .> 0)]
                    model2 = intersect(model2, names(df))
                    if length(model2) > 0
                        model = model2[:]
                    end
                end
                start = time()
                X_affine = PHD.augmentaffine(df, model=String.(model), removecols=:Constant)
                linear3, bestparams3 = PHD.regress_cv(Y, X_affine, regtype=[regtype], alpha=collect(0.1:0.1:1),
                                        missing_penalty=[1.0,2.0,4.0,6.0,8.0,12.0])
                δt = (time() - start)
                R2, OSR2 = PHD.evaluate(Y, X_affine, linear3)
                push!(results_table, [dname, SNR, k, k_missing, iter, "Affine", R2, OSR2, δt])
                CSV.write(savedir*filename, results_table)
            end
        end

        if do_finite
            println("Method Finite")
            df = deepcopy(X_missing)
            df[!,:Test] = test_ind
            start = time()
            X_missing_std = PHD.standardize(df)
            X_missing_zero_std = PHD.zeroimpute(X_missing_std)

            gm2 = PHD.trainGreedyModel(Y, X_missing_zero_std,
                                             maxdepth = 8, tolerance = 0.01, minbucket = 20, missingdata = X_missing)
            δt = (time() - start)
            R2, OSR2 = PHD.evaluate(Y, X_missing_zero_std, gm2, X_missing_std)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Finite", R2, OSR2, δt])
            CSV.write(savedir*filename, results_table)
        end
    end
end
end
