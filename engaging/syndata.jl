using Pkg
Pkg.activate("..")

using Revise
using PHD

using Random, Statistics, CSV, DataFrames, LinearAlgebra


#Generation methods
SNR = 2
model_for_y = :friedman 

p = 50
k = 20 
frac_missing = 0.5
p_missing = 0.1

missing_mar = try ARGS[2]=="1" catch; true end
# relationship_yx_mar = try ARGS[2]=="1" catch; true end

savedir = string("../results/synthetic_xmy", 
                # relationship_yx_mar ? "_mar" : "_nmar",
                # adversarial_missing ? "_adv" : "", 
                "/")
mkpath(savedir)

#Prediction methods
do_benchmark = true
do_tree = true
do_impthenreg = true
do_static = true
do_affine = true
affine_on_static_only = false #Should be set to false
do_finite = true
do_μthenreg = true 


results_main = DataFrame(dataset=[], SNR=[], k=[], kMissing=[], splitnum=[], method=[],
                                r2=[], osr2=[], time=[], hp=[])

# for ARG in ARGS
ARG = ARGS[1]
array_num = parse(Int, ARG)
d_num = mod(array_num, 71) + 1
# aux_num = div(array_num,71) + 1

dataset_list = collect(100:100:1000)
d_num = array_num + 1

n = dataset_list[d_num]#"dermatology" #"""thyroid-disease-thyroid-0387" #dataset_list[1]
@show n, k_missingsignal

Random.seed!(56465)  
X_full = generate_x(n,p, rank=round(Int, p/5))
X_missing = generate_missing(X_full; method  = missing_mar ? :mar : :censoring, p=p_missing, kmissing=round(Int,frac_missing*p))
X_full[!,:Id] .= collect(1:nrow(X_full))
X_missing[!,:Id] .= collect(1:nrow(X_missing))

canbemissing = [any(ismissing.(X_missing[:,j])) for j in names(X_missing)] #indicator of missing features

# Create output
Random.seed!(549)             
@time Y, k, k_missing = PHD.generate_y(X_full, X_missing,
                model = model_for_y,  
                k=k, k_missing_in_signal=round(Int,frac_missing*k), SNR=SNR, 
                canbemissing=canbemissing, mar=:mar)   
@show k, k_missing

test_prop = .3

savedfiles = filter(t -> startswith(t, string(dname, "_SNR_", SNR, "_nmiss_", k_missingsignal)), readdir(savedir))
map!(t -> split(replace(t, ".csv" => ""), "_")[end], savedfiles, savedfiles)
for iter in setdiff(1:10, parse.(Int, savedfiles))    
# for iter in 1:10
    @show iter
    results_table = similar(results_main,0)

    filename = string(dname, "_SNR_", SNR, "_nmiss_", k_missingsignal, "_$iter.csv")

    # Split train / test
    Random.seed!(56802+767*iter)
    # test_ind = rand(nrow(X_missing)) .< test_prop ;
    test_ind = PHD.split_dataset(X_missing, test_fraction = test_prop, random = true)
    if !random_split
        test_ind = PHD.split_dataset_nonrandom(X_missing, test_fraction = test_prop)
    end

    if do_benchmark
        println("Benchmark methods...")
        println("####################")
        d = Dict(:alpha => collect(0.1:0.1:1), :regtype => [:lasso])

        ## Method Oracle
        df = X_full[:,:]
        df[!,:Test] = test_ind
        start = time()
        linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
        δt = (time() - start)
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Oracle X", R2, OSR2, δt, bestparams[:alpha]])
        CSV.write(savedir*filename, results_table)

        df = [X_full[:,:] PHD.indicatemissing(X_missing[:,:]; removecols=:Zero)]
        df[!,:Test] = test_ind
        start = time()
        linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
        δt = (time() - start)
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Oracle XM", R2, OSR2, δt, bestparams[:alpha]])
        CSV.write(savedir*filename, results_table)


        ## Method 0
        try
            df = X_missing[:,.!canbemissing] #This step can raise an error if all features can be missing
            df[!,:Test] = test_ind
            start = time()
            linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
            δt = (time() - start)
            R2, OSR2 = PHD.evaluate(Y, df, linear)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Complete Features", R2, OSR2, δt, bestparams[:alpha]])
        catch #In this case, simply predict the mean - which leads to 0. OSR2
            push!(results_table, [dname, SNR, k, k_missing, iter, "Complete Features", 0., 0., 0.,0.])
        end
        CSV.write(savedir*filename, results_table)
    end

    if do_tree
        println("MIA-tree method...")
        println("####################")
        d = Dict(:maxdepth => collect(1:2:10))

        df = PHD.augment_MIA(X_missing)
        df[!,:Test] = test_ind
        start = time()
        cartmodel, bestparams = PHD.regress_cv(Y, df; model = :tree, parameter_dict=d)
        δt = (time() - start)
        R2, OSR2 = PHD.evaluate(Y, df, cartmodel)
        push!(results_table, [dname, SNR, k, k_missing, iter, "CART MIA", R2, OSR2, δt, bestparams[:maxdepth]])
        CSV.write(savedir*filename, results_table)
    end

    if do_impthenreg
        println("Impute-then-regress methods...")
        println("###############################")
        d = Dict(:alpha => collect(0.1:0.1:1), :regtype => [:lasso])

        ## Method 1.1
        start = time()
        X_imputed = PHD.mice_bruteforce(X_missing);
        δt = (time() - start)

        df = deepcopy(X_imputed)
        df[!,:Test] = test_ind

        start = time()
        linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
        δt += (time() - start)

        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 1", R2, OSR2, δt, bestparams[:alpha]])
        CSV.write(savedir*filename, results_table)


        ## Method 1.2
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
        linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
        δt += (time() - start)
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 2", R2, OSR2, δt, bestparams[:alpha]])
        CSV.write(savedir*filename, results_table)


        ## Method 1.3
        df = deepcopy(X_missing)
        start = time()
        X_train_imputed = PHD.mice_bruteforce(df[.!test_ind,:]);
        δt = (time() - start)

        start = time()
        X_all_imputed = PHD.mice_bruteforce(df[:,names(X_train_imputed)]);
        δt += (time() - start)

        select!(df, names(X_train_imputed))
        df[.!test_ind,:] .= X_train_imputed
        df[test_ind,:] .= X_all_imputed[test_ind,:]
        df[!,:Test] = test_ind
        start = time()
        linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
        δt += (time() - start)
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 3", R2, OSR2, δt, bestparams[:alpha]])
        CSV.write(savedir*filename, results_table)

        ## Method 1.4
        start = time()
        means_df = PHD.compute_mean(X_missing[.!test_ind,:])
        X_imputed = PHD.mean_impute(X_missing, means_df);
        δt = (time() - start)
        df = deepcopy(X_imputed)
        df[!,:Test] = test_ind
        start = time()
        linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
        δt += (time() - start)
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 4", R2, OSR2, δt, bestparams[:alpha]])
        CSV.write(savedir*filename, results_table)

        ## Method 1.5 Mean and mode impute
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
        linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
        δt += (time() - start)
        R2, OSR2 = PHD.evaluate(Y, df, linear)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 5", R2, OSR2, δt, bestparams[:alpha]])
        CSV.write(savedir*filename, results_table)
    end
    
    if do_static || do_affine
        println("Adaptive methods...")
        println("###################")
        d = Dict(:alpha => collect(0.1:0.1:1), :regtype => [:missing_weight], :missing_penalty => [1.0,2.0,4.0,6.0,8.0,12.0])


        ## Method 2: Static Adaptability
        df = deepcopy(X_missing)
        df[!,:Test] = test_ind
        start = time()
        X_augmented = hcat(PHD.zeroimpute(df), PHD.indicatemissing(df, removecols=:Zero))
        # X_augmented = PHD.zeroimpute(df)
        linear2, bestparams2 = PHD.regress_cv(Y, X_augmented, model=:linear, parameter_dict=d)
        δt = (time() - start)
        R2, OSR2 = PHD.evaluate(Y, X_augmented, linear2)
        push!(results_table, [dname, SNR, k, k_missing, iter, "Static", R2, OSR2, δt, bestparams[:alpha]])
        CSV.write(savedir*filename, results_table)

        if do_affine
            ## Method 3: Affine Adaptability
            df = deepcopy(X_missing)
            df[!,:Test] = test_ind
            model = names(df)

            start = time()
            X_affine = PHD.augmentaffine(df, model=String.(model), removecols=:Constant)
            linear3, bestparams3 = PHD.regress_cv(Y, X_affine, model=:linear, parameter_dict=d)
            δt = (time() - start)
            R2, OSR2 = PHD.evaluate(Y, X_affine, linear3)
            push!(results_table, [dname, SNR, k, k_missing, iter, "Affine", R2, OSR2, δt, bestparams[:alpha]])
            CSV.write(savedir*filename, results_table)
        end
    end
    
    if do_finite
        println("Finite adaptive methods...")
        println("###################")
        d = Dict(:maxdepth => collect(0:2:10))

        df = deepcopy(X_missing)
        df[!,:Test] = test_ind

        start = time()
        gm2, bestparams = PHD.regress_cv(Y, df, model = :greedy, parameter_dict = d)
        δt = (time() - start)
        R2, OSR2 = PHD.evaluate(Y, df, gm2)   
        push!(results_table, [dname, SNR, k, k_missing, iter, "Finite", R2, OSR2, δt, bestparams[:maxdepth]])
        CSV.write(savedir*filename, results_table)
    end
    if do_μthenreg
        println("Joint Impute-and-Regress methods...")
        println("###################")
        for model in [:linear, :tree]
            d = model == :linear ? Dict(:alpha => collect(0.1:0.1:1)) : Dict(:maxdepth => collect(1:2:10))

            df = deepcopy(X_missing)
            df[!,:Test] = test_ind

            start = time()
            opt_imp_then_reg, bestparams, μ = PHD.impute_then_regress_cv(Y, df; modeltype=model, parameter_dict=d)
            δt = (time() - start)

            R2, OSR2 = PHD.evaluate(Y, PHD.mean_impute(df, μ), opt_imp_then_reg)
            push!(results_table, [dname, SNR, k, k_missing, iter, string("Joint Imp-then-Reg - ", model), R2, OSR2, 
                        δt, model == :linear ? bestparams[:alpha] : bestparams[:maxdepth]])
            CSV.write(savedir*filename, results_table)
        end
    end
end