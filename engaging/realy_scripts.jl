println(Sys.CPU_NAME)

rm("/home/jpauph/.julia/logs/manifest_usage.toml", force=true)
rm("/home/jpauph/.julia/logs/artifact_usage.toml", force=true)

using Pkg
Pkg.activate("..")

using PHD

using Random, Statistics, CSV, DataFrames, LinearAlgebra, RCall

dataset_list = [d for d in readdir("../datasets/") if !startswith(d, ".")]
sort!(dataset_list)

savedir = string("../results/aistats-rev/", 
                "/realy", 
                "/all/")
mkpath(savedir)

#Prediction methods
do_benchmark = true
do_tree = true
do_rf_mia = true
do_impthenreg = true
do_static = true
do_affine = true
affine_on_static_only = false #Should be set to false
do_finite = true
do_μthenreg = true 
do_xgb = true 

# #Prediction methods
# do_benchmark = false
# do_tree = false
# do_rf_mia = false
# do_impthenreg = true
# do_static = false
# do_affine = false
# affine_on_static_only = false #Should be set to false
# do_finite = false
# do_μthenreg = false 
# do_xgb = false 

results_main = DataFrame(dataset=[], SNR=[], k=[], kMissing=[], splitnum=[], method=[],
                                r2=[], osr2=[], r2list=[], osr2list=[], time=[], hp=[], score=[])

function create_hp_dict(model::Symbol; small::Bool=false)
    if model == :linear 
        return Dict{Symbol,Vector}(:alpha => collect(0.1:0.1:1), :regtype => [:lasso])
    elseif model == :tree 
        return Dict{Symbol,Vector}(:maxdepth => collect(2:2:16))
    elseif model == :nn 
        return Dict{Symbol,Vector}(:hidden_nodes => collect(5:5:35))
    elseif model == :rf 
        # if small 
            return Dict{Symbol,Vector}(:ntrees => collect(50:50:200), :maxdepth => collect(10:10:50))
        # else
        #     return Dict{Symbol,Vector}(:ntrees => collect(50:25:200), :maxdepth => collect(5:5:50))
        # end
    elseif model == :adaptive 
        return Dict{Symbol,Vector}(:alpha => collect(0.1:0.1:1), :regtype => [:missing_weight], :missing_penalty => [1.0,2.0,4.0,6.0,8.0,12.0])
    elseif model == :xgboost
        # if small
            return Dict{Symbol,Vector}(:max_depth => collect(3:3:10), :min_child_weight => collect(1:2:6), :gamma => collect(0.:0.2:0.4), :n_estimators => collect(50:50:200))
        # else
        #     return Dict{Symbol,Vector}(:max_depth => collect(3:2:10), :min_child_weight => collect(1:1:6), :gamma => collect(0.:0.1:0.4), :n_estimators => collect(50:25:200))
        # end
    end
end

SNR = NaN 
k = NaN 
k_missing = 1 

# for ARG in ARGS
ARG = ARGS[1]
array_num = parse(Int, ARG)

# d_num = mod(array_num, 71) + 1
# iter_do = div(array_num, 71) + 1
d_num = array_num + 1

dname = dataset_list[d_num]#"dermatology" #"""thyroid-disease-thyroid-0387" #dataset_list[1]
@show dname

longtime_list = ["pscl-politicalInformation", "mlmRev-star"]
if  true #dname ∈ longtime_list #|| (dname == "ozone-level-detection-one" && k_missingsignal == 1)
    
    # READ INPUT: X
    X_missing = PHD.standardize_colnames(CSV.read("../datasets/"*dname*"/X_missing.csv", DataFrame, missingstrings=["", "NaN"])) #df with missing values

    # Clean up : to be checked, some datasets have strings in features
    delete_obs = PHD.string_to_float_fix!(X_missing)
    for j in PHD.unique_missing_patterns(X_missing)
        delete_obs[j] = false
    end
    X_missing = X_missing[delete_obs,:];

    #Remove intrinsic indicators
    keep_cols = names(X_missing)
    for l in values(PHD.intrinsic_indicators(X_missing, correlation_threshold=0.9))
        setdiff!(keep_cols, l)
    end
    if dname == "horse-colic"
        setdiff!(keep_cols, ["N_2"])
    end
    select!(X_missing, keep_cols)
    canbemissing = [any(ismissing.(X_missing[:,j])) for j in names(X_missing)] #indicator of missing features

    patidx, = PHD.missingness_pattern_id(X_missing)
    # READ INPUT: Y
    target_list = names(CSV.read("../datasets/"*dname*"/Y.csv", missingstrings=["", "NaN"], DataFrame))

    Y = zeros(Base.size(X_missing,1))
    if length(target_list) <= 2
        target_name = setdiff(target_list, [:Id])[1]
        Y = CSV.read("../datasets/"*dname*"/Y.csv", missingstrings=["", "NaN"], DataFrame)[findall(delete_obs),target_name]
    else
        setdiff!(target_list, [:Id, :target])
        sort!(target_list)
        Y = CSV.read("../datasets/"*dname*"/Y.csv", missingstrings=["", "NaN"], DataFrame)[findall(delete_obs),target_list[1]]
    end
    if eltype(Y) ∉ [Float64, Int64, Union{Float64,Missing}, Union{Int64,Missing}]
        using StatsBase
        cm = countmap(Y)
        level = collect(keys(cm))[argmax(collect(values(cm)))]
        Y = 1.0 .* (Y .== level)
    end
    ind_availtarget = .!ismissing.(Y)
    Y = 1.0 .* Y[ind_availtarget] #Remove missing entries before converting to Float64 !

    X_missing = X_missing[ind_availtarget,:]
    @show nrow(X_missing), ncol(X_missing), length(Y)

    patidx = patidx[ind_availtarget]
    @show length(patidx)

    if length(levels(Y)) == 2
        Y = convert(BitArray, Y)
    end

    test_prop = .3

    savedfiles = filter(t -> startswith(t, string(dname,"_real_Y_")), readdir(savedir))
    map!(t -> replace(replace(t, ".csv" => ""), string(dname,"_real_Y_") => ""), savedfiles, savedfiles)
    
    # if iter ∉ parse.(Int, savedfiles)
    for iter in setdiff(1:10, parse.(Int, savedfiles))    
    # for iter in setdiff(iter_do:iter_do, parse.(Int, savedfiles))   
    # for iter in 1:10

        # try
        @show iter
        results_table = similar(results_main,0)

        filename = string(dname, "_real_Y", "_$iter.csv")

        # Split train / test
        Random.seed!(56802+767*iter)
        test_ind = PHD.split_dataset(X_missing, test_fraction = test_prop, random = true)

        if do_xgb
            println("XGB...")
            println("####################")
            d = create_hp_dict(:xgboost)

            df = X_missing[:,:]
            df[!,:Test] = test_ind
            start = time()
            xgbmodel, bestparams, score = PHD.regress_kcv(Y, df; model = :xgboost, parameter_dict=d, stratifiedid=patidx)
            δt = (time() - start)
            R2, OSR2 = PHD.evaluate(Y, df, xgbmodel)
            R2l, OSR2l = PHD.stratified_evaluate(Y, df, xgbmodel, patidx)   
            push!(results_table, [dname, SNR, k, k_missing, iter, "XGBoost", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
        end

        if do_benchmark
            println("Benchmark methods...")
            println("####################")
            for model in [:linear, :tree, :rf]
                d = create_hp_dict(model)
                
                ## Method 0
                try
                    df = X_missing[:,.!canbemissing] #This step can raise an error if all features can be missing
                    df[!,:Test] = test_ind
                    start = time()
                    linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d, stratifiedid=patidx)
                    δt = (time() - start)
                    R2, OSR2 = PHD.evaluate(Y, df, linear)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx)   
                    push!(results_table, [dname, SNR, k, k_missing, iter, "Complete Features - $(model)", R2, OSR2, R2l, OSR2l,  δt, bestparams, score])
                catch #In this case, simply predict the mean - which leads to 0. OSR2
                    push!(results_table, [dname, SNR, k, k_missing, iter, "Complete Features - $(model)", 0., 0., [], [], 0.,Dict(), 0.])
                end
                # CSV.write(savedir*filename, results_table)
            end
        end

        if do_tree
            println("MIA-tree method...")
            println("####################")
            d = create_hp_dict(:tree)

            df = PHD.augment_MIA(X_missing)
            df[!,:Test] = test_ind
            start = time()
            cartmodel, bestparams, score = PHD.regress_kcv(Y, df; model = :tree, parameter_dict=d, stratifiedid=patidx)
            δt = (time() - start)
            R2, OSR2 = PHD.evaluate(Y, df, cartmodel)
            R2l, OSR2l = PHD.stratified_evaluate(Y, df, cartmodel, patidx)   
            push!(results_table, [dname, SNR, k, k_missing, iter, "CART MIA", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
            # CSV.write(savedir*filename, results_table)
        end

        if do_rf_mia
            println("MIA-RF method...")
            println("####################")
            d = create_hp_dict(:rf)

            df = PHD.augment_MIA(X_missing)
            df[!,:Test] = test_ind
            start = time()
            cartmodel, bestparams, score = PHD.regress_kcv(Y, df; model = :rf, parameter_dict=d, stratifiedid=patidx)
            δt = (time() - start)
            R2, OSR2 = PHD.evaluate(Y, df, cartmodel)
            R2l, OSR2l = PHD.stratified_evaluate(Y, df, cartmodel, patidx)   
            push!(results_table, [dname, SNR, k, k_missing, iter, "RF MIA", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
            # CSV.write(savedir*filename, results_table)
        end
        
        if do_impthenreg
            println("Impute-then-regress methods...")
            println("###############################")
            # for model in [:linear, :tree, :rf]
            for model in [:xgboost]
                d = create_hp_dict(model)

                ## Method 1.1
                try
                    start = time()
                    X_imputed = PHD.mice_bruteforce(X_missing);
                    
                    δt = (time() - start)
                    df = deepcopy(X_imputed)
                    df[!,:Test] = test_ind

                    start = time()
                    linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d, stratifiedid=patidx)
                    δt += (time() - start)

                    R2, OSR2 = PHD.evaluate(Y, df, linear)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx)   
                    push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 1 - $(model)", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
                    # CSV.write(savedir*filename, results_table)
                catch e
                    if isa(e, RCall.REvalError)
                        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 1 - $(model)", 0, 0, [0 for p in unique(patidx)], [0 for p in unique(patidx)], 0, Dict(), 0])
                    end
                end

                ## Method 1.2
                try
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
                    linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d, stratifiedid=patidx)
                    δt += (time() - start)
                    R2, OSR2 = PHD.evaluate(Y, df, linear)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx)   
                    push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 2 - $(model)", R2, OSR2,  R2l, OSR2l, δt, bestparams, score])
                    # CSV.write(savedir*filename, results_table)
                catch e
                    if isa(e, RCall.REvalError)
                        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 2 - $(model)", 0, 0, [0 for p in unique(patidx)], [0 for p in unique(patidx)], 0, Dict(), 0])
                    end
                end

                ## Method 1.3
                try
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
                    linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d, stratifiedid=patidx)
                    δt += (time() - start)
                    R2, OSR2 = PHD.evaluate(Y, df, linear)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx)   
                    push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 3 - $(model)", R2, OSR2,  R2l, OSR2l, δt, bestparams, score])
                    # CSV.write(savedir*filename, results_table)
                catch e
                    if isa(e, RCall.REvalError)
                        push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 3 - $(model)", 0, 0, [0 for p in unique(patidx)], [0 for p in unique(patidx)], 0, Dict(), 0])
                    end
                end
                
                ## Method 1.4
                start = time()
                means_df = PHD.compute_mean(X_missing[.!test_ind,:])
                X_imputed = PHD.mean_impute(X_missing, means_df);
                δt = (time() - start)
                df = deepcopy(X_imputed)
                df[!,:Test] = test_ind
                start = time()
                linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d, stratifiedid=patidx)
                δt += (time() - start)
                R2, OSR2 = PHD.evaluate(Y, df, linear)
                R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx)   
                push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 4 - $(model)", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
                # CSV.write(savedir*filename, results_table)

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
                linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d, stratifiedid=patidx)
                δt += (time() - start)
                R2, OSR2 = PHD.evaluate(Y, df, linear)
                R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx)   
                push!(results_table, [dname, SNR, k, k_missing, iter, "Imp-then-Reg 5 - $(model)", R2, OSR2,  R2l, OSR2l, δt, bestparams, score])
                # CSV.write(savedir*filename, results_table)
            end
        end
        
        if do_static || do_affine
            println("Adaptive methods...")
            println("###################")
            d = create_hp_dict(:adaptive)    

            ## Method 2: Static Adaptability
            df = deepcopy(X_missing)
            df[!,:Test] = test_ind
            start = time()
            X_augmented = hcat(PHD.zeroimpute(df), PHD.indicatemissing(df, removecols=:Zero))
            # X_augmented = PHD.zeroimpute(df)
            linear, bestparams, score = PHD.regress_kcv(Y, X_augmented, model=:linear, parameter_dict=d, stratifiedid=patidx)
            δt = (time() - start)
            R2, OSR2 = PHD.evaluate(Y, X_augmented, linear)
            R2l, OSR2l = PHD.stratified_evaluate(Y, X_augmented, linear, patidx)   

            push!(results_table, [dname, SNR, k, k_missing, iter, "Static", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
            # CSV.write(savedir*filename, results_table)

            if do_affine
                ## Method 3: Affine Adaptability
                df = deepcopy(X_missing)
                df[!,:Test] = test_ind
                model = names(df)

                start = time()
                X_affine = PHD.augmentaffine(df, model=String.(model), removecols=:Constant)
                linear, bestparams, score = PHD.regress_kcv(Y, X_affine, model=:linear, parameter_dict=d, stratifiedid=patidx)
                δt = (time() - start)
                R2, OSR2 = PHD.evaluate(Y, X_affine, linear)
                R2l, OSR2l = PHD.stratified_evaluate(Y, X_affine, linear, patidx)   

                push!(results_table, [dname, SNR, k, k_missing, iter, "Affine", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
                # CSV.write(savedir*filename, results_table)
            end
        end
        
        if do_finite
            println("Finite adaptive methods...")
            println("###################")
            d = create_hp_dict(:tree)

            df = deepcopy(X_missing)
            df[!,:Test] = test_ind

            start = time()
            gm2, bestparams, score = PHD.regress_kcv(Y, df, model = :greedy, parameter_dict = d, stratifiedid=patidx)
            δt = (time() - start)

            R2, OSR2 = PHD.evaluate(Y, df, gm2)  
            R2l, OSR2l = PHD.stratified_evaluate(Y, df, gm2, patidx)   
 
            push!(results_table, [dname, SNR, k, k_missing, iter, "Finite", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
            # CSV.write(savedir*filename, results_table)
        end

        if do_μthenreg
            println("Joint Impute-and-Regress methods...")
            println("###################")
            # for model in [:xgboost]
            for model in [:linear, :tree, :rf]
                d = create_hp_dict(model, small=true)
                d[:model] = [model]

                df = deepcopy(X_missing)
                df[!,:Test] = test_ind

                start = time()
                (opt_imp_then_reg, μ), bestparams, score = PHD.regress_kcv(Y, df; model=:joint, parameter_dict=d, stratifiedid=patidx)
                δt = (time() - start)

                R2, OSR2 = PHD.evaluate(Y, PHD.mean_impute(df, μ), opt_imp_then_reg)
                R2l, OSR2l = PHD.stratified_evaluate(Y, PHD.mean_impute(df, μ), opt_imp_then_reg, patidx)   

                push!(results_table, [dname, SNR, k, k_missing, iter, string("Joint Imp-then-Reg - ", model), 
                            R2, OSR2, R2l, OSR2l,
                            δt, bestparams, score])
                # CSV.write(savedir*filename, results_table)
            end
        end

        corefiles = filter(t -> startswith(t, "core."), readdir("/home/jpauph/Research/PHD/engaging/"))
        for f in corefiles
            rm("/home/jpauph/Research/PHD/engaging/"*f, force=true)
        end

        CSV.write(savedir*filename, results_table)

    end
end