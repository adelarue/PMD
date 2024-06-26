println(Sys.CPU_NAME)

rm("/home/jpauph/.julia/logs/manifest_usage.toml", force=true)
rm("/home/jpauph/.julia/logs/artifact_usage.toml", force=true)

using Pkg
Pkg.activate("..")
# Pkg.update("Flux")

# ENV["R_HOME"] = "/home/software/R/4.4.2/lib64/R/"
# ENV["PYTHON"] = ""

# using Revise
using PHD

using Random, Statistics, CSV, DataFrames, LinearAlgebra


#Generation methods
# n_list = collect(20:20:2000)
n_list = collect(20:20:1000)
# n_list = collect(1000:200:5000)
maxn = 5000

p = 10 

SNR = 2
ktotal = 5

# missingness_proba_list = collect(0.1:0.1:0.9)
missingness_proba_list = collect(0.1:0.1:0.8)

num_missing_feature = p 

random_split = true
relationship_xm_mar = try ARGS[2]=="1" catch; true end
# adversarial_missing = try ARGS[3]=="1" catch; false end
model_for_y = try ARGS[3]=="1" ? :linear : (ARGS[3]=="2" ? :tree : :nn) catch; :linear  end 

savedir = string("../results/tmlr-rev/synthetic/", 
                model_for_y,
                relationship_xm_mar ? "_mar" : "_censoring",
                "/itr_nn/")
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
do_μthenreg = false 
do_xgb = false 

function create_hp_dict(model::Symbol; small::Bool=false)
    if model == :linear 
        return Dict{Symbol,Vector}(:alpha => collect(0.1:0.1:1), :regtype => [:lasso])
    elseif model == :tree 
        return Dict{Symbol,Vector}(:maxdepth => collect(2:2:16))
    elseif model == :nn 
        return Dict{Symbol,Vector}(:hidden_nodes => collect(5:5:35))
    elseif model == :rf 
        if small 
            return Dict{Symbol,Vector}(:ntrees => collect(50:50:200), :maxdepth => collect(10:10:50))
        else
            return Dict{Symbol,Vector}(:ntrees => collect(50:25:200), :maxdepth => collect(5:5:50))
        end
    elseif model == :adaptive 
        return Dict{Symbol,Vector}(:alpha => collect(0.1:0.1:1), :regtype => [:missing_weight], :missing_penalty => [1.0,2.0,4.0,6.0,8.0,12.0])
    elseif model == :xgboost
        if small
            return Dict{Symbol,Vector}(:max_depth => collect(3:3:10), :min_child_weight => collect(1:2:6), :gamma => collect(0.:0.2:0.4), :n_estimators => collect(50:50:200))
        else
            return Dict{Symbol,Vector}(:max_depth => collect(3:2:10), :min_child_weight => collect(1:1:6), :gamma => collect(0.:0.1:0.4), :n_estimators => collect(50:25:200))
        end
    end
end

results_main = DataFrame(dataset=[], SNR=[], k=[], pMissing=[], splitnum=[], method=[],
                                r2=[], osr2=[],
                                r2list=[], osr2list=[], 
                                muvec =[],
                                time=[], hp=[], score=[])


ARG = ARGS[1]
array_num = parse(Int, ARG)
# for array_num in 1:length(n_list)

    n = n_list[array_num+1]
    dname = string("n_", n, "_p_", p)

    @show model_for_y, n


    # Create output
    Random.seed!(565)
    X_full = PHD.generate_x(maxn+5000, p; rank=floor(Int, p/2))
        
    # @time Y, k, k_missing = PHD.generate_y(X_full, X_full,
    #                 model = model_for_y,  
    #                 k=ktotal, k_missing_in_signal=0, SNR=SNR, 
    #                 mar=true)   

    # @show k, k_missing

    test_ind = BitArray(vec([zeros(n)' ones(5000)']))

    # @show Base.size(X_full), mean(Y)
    
    for aux_num in 1:length(missingness_proba_list)
        missingness_proba = missingness_proba_list[aux_num]
        @show missingness_proba
        savedfiles = filter(t -> startswith(t, string("n_", n, "_p_", p, "_pmiss_", missingness_proba)), readdir(savedir))
        map!(t -> split(replace(t, ".csv" => ""), "_")[end], savedfiles, savedfiles)


        for iter in setdiff(1:10, parse.(Int, savedfiles)) 
            Random.seed!(565)
            X_missing = PHD.generate_missing(X_full; 
                        method = relationship_xm_mar ? :mar : :censoring, 
                        p=missingness_proba, 
                        kmissing=num_missing_feature)

            Random.seed!(565+mod(iter-1,5)*3467)                
            @time Y, k, k_missing = PHD.generate_y(X_full, X_full,
                            model = model_for_y,  
                            k=ktotal, k_missing_in_signal=0, SNR=SNR, 
                            mar=true)   
        
            @show k, k_missing
        # for iter in 1:10
            Random.seed!(565+div(iter-1,5)*7)
            # X_missing = PHD.generate_missing(X_full; 
            #             method = relationship_xm_mar ? :mar : :censoring, 
            #             p=missingness_proba, 
            #             kmissing=num_missing_feature)
            
            selectobs = shuffle(1:Base.size(X_full, 1))[1:(5000+n)]
            global X_full = X_full[selectobs,:] 
            X_missing = X_missing[selectobs,:] 
            global Y = Y[selectobs] 

            @show iter
            results_table = similar(results_main,0)

            filename = string("n_", n, "_p_", p, "_pmiss_", missingness_proba, "_$iter.csv")

            X_full[!,:Id] .= collect(1:nrow(X_full))
            X_missing[!,:Id] .= collect(1:nrow(X_missing))

            for n in names(X_missing)
                for i in 1:nrow(X_missing)
                    if ismissing(X_missing[i,n])
                        ()
                    elseif isnan(X_missing[i,n])
                        X_missing[i,n] = missing
                    end
                end
            end
            
            canbemissing = [j for j in names(X_missing) if any(ismissing.(X_missing[:,j]))] #indicator of missing features
            patidx, = PHD.missingness_pattern_id(X_missing)
            subsetpattern = unique(patidx[1:n])
            
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
                R2l, OSR2l = PHD.stratified_evaluate(Y, df, xgbmodel, patidx, subsetpattern=subsetpattern)   
                push!(results_table, [dname, SNR, k, missingness_proba, iter, "XGBoost", R2, OSR2, R2l, OSR2l, [], δt, bestparams, score])
            end

            if do_benchmark
                println("Benchmark methods...")
                println("####################")
                bench_model = model_for_y == :nn ? :rf : model_for_y
                d = create_hp_dict(bench_model)

                ## Method Oracle
                df = X_full[:,:]
                df[!,:Test] = test_ind
                start = time()
                linear, bestparams, score = PHD.regress_kcv(Y, df, model=bench_model, parameter_dict=d)
                δt = (time() - start)
                R2, OSR2 = PHD.evaluate(Y, df, linear)
                R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx, subsetpattern=subsetpattern)   
                push!(results_table, [dname, SNR, k, missingness_proba, iter, "Oracle X", R2, OSR2, R2l, OSR2l, [], δt, bestparams, score])
                # CSV.write(savedir*filename, results_table)

                df = [X_full[:,:] PHD.indicatemissing(X_missing[:,:]; removecols=:Zero)]
                df[!,:Test] = test_ind
                start = time()
                linear, bestparams, score = PHD.regress_kcv(Y, df, model=bench_model, parameter_dict=d)
                δt = (time() - start)
                R2, OSR2 = PHD.evaluate(Y, df, linear)
                R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx, subsetpattern=subsetpattern)   
                push!(results_table, [dname, SNR, k, missingness_proba, iter, "Oracle XM", R2, OSR2, R2l, OSR2l, [], δt, bestparams, score])
                # CSV.write(savedir*filename, results_table)


                ## Method 0
                try
                    df = X_missing[:,.!canbemissing] #This step can raise an error if all features can be missing
                    df[!,:Test] = test_ind
                    start = time()
                    linear, bestparams, score = PHD.regress_kcv(Y, df, model=bench_model, parameter_dict=d)
                    δt = (time() - start)
                    R2, OSR2 = PHD.evaluate(Y, df, linear)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx, subsetpattern=subsetpattern)   
                    push!(results_table, [dname, SNR, k, missingness_proba, iter, "Complete Features", R2, OSR2, R2l, OSR2l, [], δt, bestparams, score])
                catch #In this case, simply predict the mean - which leads to 0. OSR2
                    push!(results_table, [dname, SNR, k, missingness_proba, iter, "Complete Features", 0., 0., [], [], [], 0.,0., 0.])
                end
                # CSV.write(savedir*filename, results_table)
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
                R2l, OSR2l = PHD.stratified_evaluate(Y, df, cartmodel, patidx, subsetpattern=subsetpattern)   
                push!(results_table, [dname, SNR, k, missingness_proba, iter, "CART MIA", R2, OSR2, R2l, OSR2l, [], δt, bestparams, score])
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
                push!(results_table, [dname, SNR, k, missingness_proba, iter, "RF MIA", R2, OSR2, R2l, OSR2l, [], δt, bestparams, score])
                # CSV.write(savedir*filename, results_table)
            end
            if do_impthenreg
                # for model in [:xgboost, :linear, :tree, :rf]
                for model in [:nn]
                    println("Impute-then-regress methods...")
                    println("###############################")
                    d = create_hp_dict(model)

                    ## Method 1.1: Mice on train+test
                    start = time()
                    X_imputed = PHD.mice_bruteforce(X_missing);
                    δt = (time() - start)

                    df = deepcopy(X_imputed)
                    df[!,:Test] = test_ind

                    start = time()
                    linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d)
                    δt += (time() - start)

                    R2, OSR2 = PHD.evaluate(Y, df, linear)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx, subsetpattern=subsetpattern)   

                    push!(results_table, [dname, SNR, k, missingness_proba, iter, "Imp-then-Reg 1 - $(model)", R2, OSR2,  R2l, OSR2l, [], δt, bestparams, score])
                    # CSV.write(savedir*filename, results_table)


                    ## Method 1.2: Mice on train + test with imputed train
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
                    linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d)
                    δt += (time() - start)
                    R2, OSR2 = PHD.evaluate(Y, df, linear)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx, subsetpattern=subsetpattern)   

                    push!(results_table, [dname, SNR, k, missingness_proba, iter, "Imp-then-Reg 2 - $(model)", R2, OSR2,  R2l, OSR2l, [], δt, bestparams, score])
                    # CSV.write(savedir*filename, results_table)


                    ## Method 1.3: Mice on train + test with original train
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
                    linear, bestparams, score = PHD.regress_kcv(Y, df, model=model, parameter_dict=d)
                    δt += (time() - start)
                    R2, OSR2 = PHD.evaluate(Y, df, linear)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx, subsetpattern=subsetpattern)   
                    push!(results_table, [dname, SNR, k, missingness_proba, iter, "Imp-then-Reg 3 - $(model)", R2, OSR2, R2l, OSR2l, [], δt, bestparams, score])
                    # CSV.write(savedir*filename, results_table)

                    ## Method 1.4: Mean-impute then regress
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
                    R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx, subsetpattern=subsetpattern)   
                    push!(results_table, [dname, SNR, k, missingness_proba, iter, "Imp-then-Reg 4 - $(model)", R2, OSR2, R2l, OSR2l, 
                            Matrix(means_df[:,canbemissing])[1,:], δt, bestparams, score])
                    # CSV.write(savedir*filename, results_table)

                    # ## Method 1.5 Mean and mode impute
                    # start = time()
                    # means_df = PHD.compute_mean(X_missing[.!test_ind,:])
                    # X_imputed = PHD.mean_impute(X_missing, means_df);
                    # δt = (time() - start)
                    # df = deepcopy(X_imputed)
                    # start = time()
                    # PHD.mode_impute!(df, train = .!test_ind)
                    # δt += (time() - start)
                    # df[!,:Test] = test_ind
                    # start = time()
                    # linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
                    # δt += (time() - start)
                    # R2, OSR2 = PHD.evaluate(Y, df, linear)
                    # push!(results_table, [dname, SNR, k, missingness_proba, iter, "Imp-then-Reg 5", R2, OSR2, δt, bestparams[:alpha]])
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
                    R2l, OSR2l = PHD.stratified_evaluate(Y, X_augmented, linear, patidx, subsetpattern=subsetpattern)  
                    μ = PHD.recover_mu(linear, canbemissing) 
                    push!(results_table, [dname, SNR, k, missingness_proba, iter, "Static", R2, OSR2, R2l, OSR2l, μ, δt, bestparams, score])
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
                    R2l, OSR2l = PHD.stratified_evaluate(Y, X_affine, linear, patidx, subsetpattern=subsetpattern)  

                    push!(results_table, [dname, SNR, k, missingness_proba, iter, "Affine", R2, OSR2, R2l, OSR2l, [], δt, bestparams, score])
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
                R2l, OSR2l = PHD.stratified_evaluate(Y, df, gm2, patidx, subsetpattern=subsetpattern)  
  
                push!(results_table, [dname, SNR, k, missingness_proba, iter, "Finite", R2, OSR2, R2l, OSR2l, [], δt, bestparams, score])
                # CSV.write(savedir*filename, results_table)
            end

            if do_μthenreg
                println("Joint Impute-and-Regress methods...")
                println("###################")
                for model in [:linear, :tree, :rf]
                    # d = Dict(:maxdepth => collect(6:2:10))
                    # d = model == :linear ? Dict(:alpha => collect(0.1:0.1:1)) : Dict(:maxdepth => collect(1:2:10))
                    d = create_hp_dict(model)
                    d[:model] = [model]

                    df = deepcopy(X_missing)
                    df[!,:Test] = test_ind

                    start = time()
                    (opt_imp_then_reg, μ), bestparams, score = PHD.regress_kcv(Y, df; model=:joint, parameter_dict=d, stratifiedid=patidx)
                    δt = (time() - start)

                    R2, OSR2 = PHD.evaluate(Y, PHD.mean_impute(df, μ), opt_imp_then_reg)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, PHD.mean_impute(df, μ), opt_imp_then_reg, patidx, subsetpattern=subsetpattern)   
                    push!(results_table, [dname, SNR, k, missingness_proba, iter, string("Joint Imp-then-Reg - ", model), R2, OSR2, R2l, OSR2l,
                                Matrix(μ[:,canbemissing])[1,:], 
                                δt, model == :linear ? bestparams : bestparams, score])
                    # CSV.write(savedir*filename, results_table)
                end
            end

            # corefiles = filter(t -> startswith(t, "core."), readdir("/home/jpauph/Research/PHD/engaging/"))
            # for f in corefiles
            #     rm("/home/jpauph/Research/PHD/engaging/"*f, force=true)
            # end

            CSV.write(savedir*filename, results_table)
        end
    end 
# end