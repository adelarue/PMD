# println(Sys.CPU_NAME)

rm("/home/jpauph/.julia/logs/manifest_usage.toml", force=true)
rm("/home/jpauph/.julia/logs/artifact_usage.toml", force=true)

using Pkg
Pkg.activate("..")

ENV["R_HOME"] = "/home/software/R/4.4.2/lib64/R"
ENV["PYTHON"] = ""
# @show haskey(ENV, "PYTHONFAULTHANDLER")
# ENV["PYTHONFAULTHANDLER"] = ""
# using Revise
using PHD

using Random, Statistics, CSV, DataFrames, LinearAlgebra, RCall

dataset_list = [d for d in readdir("../datasets/") if !startswith(d, ".")]
sort!(dataset_list)

pb_datasets = ["cylinder-bands", "ozone-level-detection-eight", "ozone-level-detection-one", "thyroid-disease-thyroid-0387", "trains"]

missingsignal_list = [0,1,2,3,4,5,6,7,8,9,10]
# missingsignal_list = [0,1,2,3,4,5]

# missingsignal_list = [-1]


#Generation methods
SNR = 2
ktotal = 10
random_split = true
relationship_yx_mar = try ARGS[2]=="1" catch; true end
adversarial_missing = try ARGS[3]=="1" catch; false end
model_for_y = try Symbol(ARGS[4]) catch ; :linear end
# model_for_y = :nn 

# savedir = string("../results/aistats-rev/fakey/", 
#                 model_for_y,
#                 relationship_yx_mar ? "_mar" : "_nmar",
#                 adversarial_missing ? "_adv" : "", 
#                 "/all/")
# mkpath(savedir)

# #Prediction methods
# do_benchmark = true
# do_tree = true
# do_rf_mia = true
# do_impthenreg = true
# do_static = true
# do_affine = true
# affine_on_static_only = false #Should be set to false
# do_finite = true
# do_μthenreg = true 
# do_xgb = true

savedir = string("../results/aistats-rev/fakey/", 
                model_for_y,
                relationship_yx_mar ? "_mar" : "_nmar",
                adversarial_missing ? "_adv" : "", 
                "/itr/")
mkpath(savedir)

#Prediction methods
do_benchmark = false
do_tree = false
do_rf_mia = false
do_impthenreg = true
do_static = false
do_affine = false
affine_on_static_only = false #Should be set to false
do_finite = false
do_μthenreg = true 
do_xgb = false

function create_hp_dict(model::Symbol; small::Bool=false)
    if model == :linear 
        if small
            return Dict{Symbol,Vector}(:alpha => collect(0.1:0.3:1), :regtype => [:lasso])
        else
            return Dict{Symbol,Vector}(:alpha => collect(0.1:0.1:1), :regtype => [:lasso])
        end
    elseif model == :tree 
        return Dict{Symbol,Vector}(:maxdepth => collect(2:2:16))
    elseif model == :nn 
        return Dict{Symbol,Vector}(:hidden_nodes => collect(5:5:35))
    elseif model == :rf 
        if small 
            return Dict{Symbol,Vector}(:ntrees => collect(100:100:200), :maxdepth => collect(10:20:50))
        else
            return Dict{Symbol,Vector}(:ntrees => collect(50:50:200), :maxdepth => collect(10:10:50))
        end
    elseif model == :adaptive 
        return Dict{Symbol,Vector}(:alpha => collect(0.1:0.1:1), :regtype => [:missing_weight], :missing_penalty => [1.0,2.0,4.0,6.0,8.0,12.0])
    elseif model == :xgboost
        if small
            return Dict{Symbol,Vector}(:max_depth => collect(3:5:10), :gamma => collect(0.:0.2:0.4), :n_estimators => collect(100:100:200))
        else
            return Dict{Symbol,Vector}(:max_depth => collect(3:3:10), :min_child_weight => collect(1:2:6), :gamma => collect(0.:0.2:0.4), :n_estimators => collect(50:50:200))
        end
    end
end
results_main = DataFrame(dataset=[], SNR=[], k=[], kMissing=[], splitnum=[], method=[],
                                r2=[], osr2=[], 
                                r2list=[], osr2list=[], 
                                time=[], hp=[], score=[])

# for ARG in ARGS
ARG = ARGS[1]
array_num = parse(Int, ARG)

# d_num = mod(array_num, 71) + 1
# iter_do = div(array_num,71) + 1

d_num = array_num + 1

for aux_num in 1:length(missingsignal_list)

    dname = dataset_list[d_num]#"dermatology" #"""thyroid-disease-thyroid-0387" #dataset_list[1]
    k_missingsignal = missingsignal_list[aux_num]
    @show dname, k_missingsignal

    # longtime_list = ["pscl-politicalInformation", "mlmRev-star"]
    if  dname ∉ pb_datasets #dname ∈ longtime_list #|| (dname == "ozone-level-detection-one" && k_missingsignal == 1)
        # Read in a data file.
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

        # X_full = PHD.standardize_colnames(CSV.read("../datasets/"*dname*"/X_full.csv", DataFrame))[delete_obs,keep_cols] #ground truth df
        # PHD.string_to_float_fix!(X_full)
        X_full = PHD.standardize_colnames(CSV.read("../datasets/"*dname*"/X_full.csv", DataFrame))[delete_obs,:] #ground truth df
        PHD.string_to_float_fix!(X_full)
        X_full = X_full[:,keep_cols]

        if adversarial_missing
            # optimize missingness pattern for outlier suppression
            X_missing = PHD.optimize_missingness(X_missing, X_full)
        end

        patidx, = PHD.missingness_pattern_id(X_missing)

        test_prop = .3

        
        savedfiles = filter(t -> startswith(t, string(dname, "_SNR_", SNR, "_nmiss_", k_missingsignal)), readdir(savedir))
        map!(t -> split(replace(t, ".csv" => ""), "_")[end], savedfiles, savedfiles)

        # for iter in setdiff(iter_do:iter_do, parse.(Int, savedfiles))    
        for iter in setdiff(1:10, parse.(Int, savedfiles))    
            @show iter

            # Create output
            Random.seed!(565+mod(iter-1,5)*47)             
            @time Y, k, k_missing = PHD.generate_y(X_full, X_missing,
                            model = model_for_y,  
                            k=ktotal, k_missing_in_signal=k_missingsignal, SNR=SNR, 
                            canbemissing=canbemissing, mar=relationship_yx_mar)   
            @show k, k_missing


             if k_missing == k_missingsignal #If not enough missing features to generate Y with k_missingsignal, abort (already done)

            # for iter in 1:10
                results_table = similar(results_main,0)

                filename = string(dname, "_SNR_", SNR, "_nmiss_", k_missingsignal, "_$iter.csv")

                # Split train / test
                Random.seed!(56802+767*div(iter-1,5))
                # test_ind = rand(nrow(X_missing)) .< test_prop ;
                test_ind = PHD.split_dataset(X_missing, test_fraction = test_prop, random = true)
                # if !random_split
                # 	test_ind = PHD.split_dataset_nonrandom(X_missing, test_fraction = test_prop)
                # end

                if do_xgb
                    println("XGB...")
                    println("####################")
                    d = create_hp_dict(:xgboost)

                    df = X_missing[:,:]
                    df[!,:Test] = test_ind
                    start = time()
                    @time xgbmodel, bestparams, score = PHD.regress_kcv(Y, df; model = :xgboost, parameter_dict=d, stratifiedid=patidx)
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
                        
                        ## Method Oracle
                        df = X_full[:,:]
                        df[!,:Test] = test_ind
                        start = time()
                        linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d)
                        δt = (time() - start)
                        R2, OSR2 = PHD.evaluate(Y, df, linear)
                        R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx)   
                        push!(results_table, [dname, SNR, k, k_missing, iter, "Oracle X - $(model)", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
                        # CSV.write(savedir*filename, results_table)

                        df = [X_full[:,:] PHD.indicatemissing(X_missing[:,:]; removecols=:Zero)]
                        df[!,:Test] = test_ind
                        start = time()
                        linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d)
                        δt = (time() - start)
                        R2, OSR2 = PHD.evaluate(Y, df, linear)
                        R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx)   
                        push!(results_table, [dname, SNR, k, k_missing, iter, "Oracle XM - $(model)", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
                        # CSV.write(savedir*filename, results_table)

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
                    @time cartmodel, bestparams, score = PHD.regress_kcv(Y, df; model = :tree, parameter_dict=d, stratifiedid=patidx)
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
                    @time cartmodel, bestparams, score = PHD.regress_kcv(Y, df; model = :rf, parameter_dict=d, stratifiedid=patidx)
                    δt = (time() - start)
                    R2, OSR2 = PHD.evaluate(Y, df, cartmodel)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, cartmodel, patidx)   
                    push!(results_table, [dname, SNR, k, k_missing, iter, "RF MIA", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
                    # CSV.write(savedir*filename, results_table)
                end

                if do_impthenreg
                    println("Impute-then-regress methods...")
                    println("###############################")
                    for model in [:linear, :tree, :rf]
                    # for model in [:xgboost]
                        d = create_hp_dict(model)
        
                        ## Method 1.1
                        try
                            start = time()
                            X_imputed = PHD.mice_bruteforce(X_missing);
                            δt = (time() - start)
                            df = deepcopy(X_imputed)
                            df[!,:Test] = test_ind
                            start = time()
                            @time linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d, stratifiedid=patidx)
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
    
                    if do_static 
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
                        # μ = PHD.recover_mu(linear, canbemissing) 
                        push!(results_table, [dname, SNR, k, k_missing, iter, "Static", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
                        # CSV.write(savedir*filename, results_table)
                    end 
    
                    if do_affine
                        ## Method 3: Affine Adaptability
                        df = deepcopy(X_missing)
                        df[!,:Test] = test_ind
                        model = names(df)
                        # if affine_on_static_only
                        #     model2 = names(linear2)[findall(abs.([linear2[1,c] for c in names(linear2)]) .> 0)]
                        #     model2 = intersect(model2, names(df))
                        #     if length(model2) > 0
                        #         model = model2[:]
                        #     end
                        # end
                        start = time()
                        X_affine = PHD.augmentaffine(df, model=String.(model), removecols=:Constant)
                        linear, bestparams, score = PHD.regress_kcv(Y, X_affine, model=:linear, parameter_dict=d)
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
                    gm2, bestparams, score = PHD.regress_kcv(Y, df, model = :greedy, parameter_dict = d)
                    δt = (time() - start)
                    R2, OSR2 = PHD.evaluate(Y, df, gm2) 
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, gm2, patidx)  
      
                    push!(results_table, [dname, SNR, k, k_missing, iter, "Finite", R2, OSR2, R2l, OSR2l, δt, bestparams, score])
                    # CSV.write(savedir*filename, results_table)
                end

                if do_μthenreg
                    println("Joint Impute-and-Regress methods...")
                    println("###################")
                    for model in [:xgboost]
                    # for model in [:linear, :tree, :rf]
                        d = create_hp_dict(model, small=true)
                        d[:model] = [model]

                        df = deepcopy(X_missing)
                        df[!,:Test] = test_ind

                        start = time()
                        @time (opt_imp_then_reg, μ), bestparams, score = PHD.regress_kcv(Y, df; model=:joint, parameter_dict=d, stratifiedid=patidx)
                        δt = (time() - start)

                        R2, OSR2 = PHD.evaluate(Y, PHD.mean_impute(df, μ), opt_imp_then_reg)
                        R2l, OSR2l = PHD.stratified_evaluate(Y, PHD.mean_impute(df, μ), opt_imp_then_reg, patidx)   
                        push!(results_table, [dname, SNR, k, k_missing, iter, string("Joint Imp-then-Reg - ", model), R2, OSR2, R2l, OSR2l,
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
    end
end 
# end
