using Pkg
Pkg.activate("..")

# using Revise
using PHD

using Random, Statistics, CSV, DataFrames, LinearAlgebra


#Generation methods
# n_list = collect(20:20:2000)
n_list = collect(20:20:1000)

p = 10 

SNR = 2
ktotal = 5

# missingness_proba_list = collect(0.05:0.05:0.9)
missingness_proba_list = collect(0.1:0.1:0.9)
# missingness_proba_list = collect(0.2:0.1:0.9)

# missingness_proba = 0.1 
num_missing_feature = p 

random_split = true
relationship_xm_mar = try ARGS[2]=="1" catch; true end
# adversarial_missing = try ARGS[3]=="1" catch; false end
model_for_y = try ARGS[3]=="1" ? :linear : (ARGS[3]=="2" ? :tree : :nn) catch; :linear  end 

savedir = string("../results/synthetic/", 
                model_for_y,
                relationship_xm_mar ? "_mar" : "_censoring",
                "/comparison/")
mkpath(savedir)

#Prediction methods
# do_benchmark = true
do_tree = model_for_y == :tree
do_impthenreg = true
do_static = true
# do_affine = true
# affine_on_static_only = false #Should be set to false
# do_finite = true
do_μthenreg = true 

function create_hp_dict(model::Symbol)
    if model == :linear 
        return Dict{Symbol,Vector}(:alpha => collect(0.1:0.1:1), :regtype => [:lasso])
    elseif model == :tree 
        return Dict{Symbol,Vector}(:maxdepth => collect(0:2:10))
    elseif model == :nn 
        return Dict{Symbol,Vector}(:hidden_nodes => collect(5:5:35))
    elseif model == :rf 
        return Dict{Symbol,Vector}(:ntrees => collect(20:20:100), :maxdepth => collect(5:5:20))
    elseif model == :adaptive 
        return Dict{Symbol,Vector}(:alpha => collect(0.1:0.1:1), :regtype => [:missing_weight], :missing_penalty => [1.0,2.0,4.0,6.0,8.0,12.0])
    end
end

results_main = DataFrame(dataset=[], SNR=[], k=[], pMissing=[], splitnum=[], method=[],
                                r2=[], osr2=[],
                                r2list=[], osr2list=[], 
                                muvec =[],
                                time=[], hp=[])


ARG = ARGS[1]
array_num = parse(Int, ARG)
# for array_num in 1:length(n_list)

    n = n_list[array_num+1]
    dname = string("n_", n, "_p_", p)

    @show n


    # Create output
    Random.seed!(549)
    X_full = PHD.generate_x(15000, p; rank=floor(Int, p/2))[union(1:n, 10001:15000),:]
        
    @time Y, k, k_missing = PHD.generate_y(X_full, X_full,
                    model = model_for_y,  
                    k=ktotal, k_missing_in_signal=0, SNR=SNR, 
                    mar=true)   

    @show k, k_missing

    test_ind = BitArray(vec([zeros(n)' ones(5000)']))

    @show mean(Y)
    
    for aux_num in 1:length(missingness_proba_list)
        missingness_proba = missingness_proba_list[aux_num]
        @show missingness_proba
        savedfiles = filter(t -> startswith(t, string("n_", n, "_p_", p, "_pmiss_", missingness_proba)), readdir(savedir))
        map!(t -> split(replace(t, ".csv" => ""), "_")[end], savedfiles, savedfiles)
        
        # for iter in setdiff(1:10, parse.(Int, savedfiles))    
        for iter in 1:10
            Random.seed!(549+n*5131+iter*7)
            X_missing = PHD.generate_missing(X_full; 
                        method = relationship_xm_mar ? :mar : :censoring, 
                        p=missingness_proba, 
                        kmissing=num_missing_feature)

            @show iter
            results_table = similar(results_main,0)

            filename = string("n_", n, "_p_", p, "_pmiss_", missingness_proba, "_$iter.csv")

            X_full[!,:Id] .= collect(1:nrow(X_full))
            X_missing[!,:Id] .= collect(1:nrow(X_missing))

            canbemissing = [j for j in names(X_missing) if any(ismissing.(X_missing[:,j]))] #indicator of missing features
            patidx, = PHD.missingness_pattern_id(X_missing)
            subsetpattern = unique(patidx[1:n])

            # Split train / test

            # if do_benchmark
            #     println("Benchmark methods...")
            #     println("####################")
            #     d = Dict(:alpha => collect(0.1:0.1:1), :regtype => [:lasso])

            #     ## Method Oracle
            #     df = X_full[:,:]
            #     df[!,:Test] = test_ind
            #     start = time()
            #     linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
            #     δt = (time() - start)
            #     R2, OSR2 = PHD.evaluate(Y, df, linear)
            #     push!(results_table, [dname, SNR, k, missingness_proba, iter, "Oracle X", R2, OSR2, δt, bestparams[:alpha]])
            #     CSV.write(savedir*filename, results_table)

            #     df = [X_full[:,:] PHD.indicatemissing(X_missing[:,:]; removecols=:Zero)]
            #     df[!,:Test] = test_ind
            #     start = time()
            #     linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
            #     δt = (time() - start)
            #     R2, OSR2 = PHD.evaluate(Y, df, linear)
            #     push!(results_table, [dname, SNR, k, missingness_proba, iter, "Oracle XM", R2, OSR2, δt, bestparams[:alpha]])
            #     CSV.write(savedir*filename, results_table)


            #     ## Method 0
            #     try
            #         df = X_missing[:,.!canbemissing] #This step can raise an error if all features can be missing
            #         df[!,:Test] = test_ind
            #         start = time()
            #         linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
            #         δt = (time() - start)
            #         R2, OSR2 = PHD.evaluate(Y, df, linear)
            #         push!(results_table, [dname, SNR, k, missingness_proba, iter, "Complete Features", R2, OSR2, δt, bestparams[:alpha]])
            #     catch #In this case, simply predict the mean - which leads to 0. OSR2
            #         push!(results_table, [dname, SNR, k, missingness_proba, iter, "Complete Features", 0., 0., 0.,0.])
            #     end
            #     CSV.write(savedir*filename, results_table)
            # end

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
                R2l, OSR2l = PHD.stratified_evaluate(Y, df, cartmodel, patidx, subsetpattern=subsetpattern)   
                push!(results_table, [dname, SNR, k, missingness_proba, iter, "CART MIA", R2, OSR2, R2l, OSR2l, [], δt, bestparams[:maxdepth]])
                CSV.write(savedir*filename, results_table)
            end

            if do_impthenreg
                println("Impute-then-regress methods...")
                println("###############################")
                d = model_for_y == :linear ? 
                        Dict(:alpha => collect(0.1:0.1:1), :regtype => [:lasso]) :
                        (model_for_y == :tree ? Dict(:maxdepth => collect(1:2:10)) : Dict(:hidden_nodes => collect(5:5:20)) )


                # ## Method 1.1
                # start = time()
                # X_imputed = PHD.mice_bruteforce(X_missing);
                # δt = (time() - start)

                # df = deepcopy(X_imputed)
                # df[!,:Test] = test_ind

                # start = time()
                # linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
                # δt += (time() - start)

                # R2, OSR2 = PHD.evaluate(Y, df, linear)
                # push!(results_table, [dname, SNR, k, missingness_proba, iter, "Imp-then-Reg 1", R2, OSR2, δt, bestparams[:alpha]])
                # CSV.write(savedir*filename, results_table)


                # ## Method 1.2
                # df = deepcopy(X_missing)
                # start = time()
                # X_train_imputed = PHD.mice_bruteforce(df[.!test_ind,:]);
                # δt = (time() - start)

                # select!(df, names(X_train_imputed))
                # df[.!test_ind,:] .= X_train_imputed
                # start = time()
                # X_all_imputed = PHD.mice_bruteforce(df);
                # δt += (time() - start)

                # df = deepcopy(X_all_imputed)
                # df[!,:Test] = test_ind

                # start = time()
                # linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
                # δt += (time() - start)
                # R2, OSR2 = PHD.evaluate(Y, df, linear)
                # push!(results_table, [dname, SNR, k, missingness_proba, iter, "Imp-then-Reg 2", R2, OSR2, δt, bestparams[:alpha]])
                # CSV.write(savedir*filename, results_table)


                # ## Method 1.3
                # df = deepcopy(X_missing)
                # start = time()
                # X_train_imputed = PHD.mice_bruteforce(df[.!test_ind,:]);
                # δt = (time() - start)

                # start = time()
                # X_all_imputed = PHD.mice_bruteforce(df[:,names(X_train_imputed)]);
                # δt += (time() - start)

                # select!(df, names(X_train_imputed))
                # df[.!test_ind,:] .= X_train_imputed
                # df[test_ind,:] .= X_all_imputed[test_ind,:]
                # df[!,:Test] = test_ind
                # start = time()
                # linear, bestparams = PHD.regress_cv(Y, df, model=:linear, parameter_dict=d)
                # δt += (time() - start)
                # R2, OSR2 = PHD.evaluate(Y, df, linear)
                # push!(results_table, [dname, SNR, k, missingness_proba, iter, "Imp-then-Reg 3", R2, OSR2, δt, bestparams[:alpha]])
                # CSV.write(savedir*filename, results_table)

                ## Method 1.4
                start = time()
                means_df = PHD.compute_mean(X_missing[.!test_ind,:])
                X_imputed = PHD.mean_impute(X_missing, means_df);
                δt = (time() - start)
                df = deepcopy(X_imputed)
                df[!,:Test] = test_ind
                start = time()
                linear, bestparams = PHD.regress_cv(Y, df, model=model_for_y, parameter_dict=d)
                δt += (time() - start)
                R2, OSR2 = PHD.evaluate(Y, df, linear)
                R2l, OSR2l = PHD.stratified_evaluate(Y, df, linear, patidx, subsetpattern=subsetpattern)   
                push!(results_table, [dname, SNR, k, missingness_proba, iter, "Imp-then-Reg 4", R2, OSR2, R2l, OSR2l, 
                        Matrix(means_df[:,canbemissing])[1,:], δt, model_for_y == :linear ?  bestparams[:alpha] : bestparams[:maxdepth]])
                CSV.write(savedir*filename, results_table)

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
            
            if do_static #|| do_affine
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
                R2l, OSR2l = PHD.stratified_evaluate(Y, X_augmented, linear2, patidx, subsetpattern=subsetpattern)  
                μ = PHD.recover_mu(linear2, canbemissing) 
                push!(results_table, [dname, SNR, k, missingness_proba, iter, "Static", R2, OSR2, R2l, OSR2l, μ, δt, bestparams2[:alpha]])
                CSV.write(savedir*filename, results_table)

                # if do_affine
                #     ## Method 3: Affine Adaptability
                #     df = deepcopy(X_missing)
                #     df[!,:Test] = test_ind
                #     model = names(df)
                #     # if affine_on_static_only
                #     #     model2 = names(linear2)[findall(abs.([linear2[1,c] for c in names(linear2)]) .> 0)]
                #     #     model2 = intersect(model2, names(df))
                #     #     if length(model2) > 0
                #     #         model = model2[:]
                #     #     end
                #     # end
                #     start = time()
                #     X_affine = PHD.augmentaffine(df, model=String.(model), removecols=:Constant)
                #     linear3, bestparams3 = PHD.regress_cv(Y, X_affine, model=:linear, parameter_dict=d)
                #     δt = (time() - start)
                #     R2, OSR2 = PHD.evaluate(Y, X_affine, linear3)
                #     push!(results_table, [dname, SNR, k, missingness_proba, iter, "Affine", R2, OSR2, δt, bestparams3[:alpha]])
                #     CSV.write(savedir*filename, results_table)
                # end
            end
            
            # if do_finite
            #     println("Finite adaptive methods...")
            #     println("###################")
            #     d = Dict(:maxdepth => collect(0:2:10))

            #     df = deepcopy(X_missing)
            #     df[!,:Test] = test_ind

            #     start = time()
            #     # X_missing_std = PHD.standardize(df)
            #     # X_missing_std = deepcopy(df)
            #     # X_missing_zero_std = PHD.zeroimpute(X_missing_std)
            #     gm2, bestparams = PHD.regress_cv(Y, df, model = :greedy, parameter_dict = d)
            #     δt = (time() - start)
            #     @show R2, OSR2 = PHD.evaluate(Y, df, gm2)   
            #     push!(results_table, [dname, SNR, k, missingness_proba, iter, "Finite", R2, OSR2, δt, bestparams[:maxdepth]])
            #     CSV.write(savedir*filename, results_table)
            # end
            if do_μthenreg
                println("Joint Impute-and-Regress methods...")
                println("###################")
                for model in [model_for_y, :rf]
                    # d = Dict(:maxdepth => collect(6:2:10))
                    # d = model == :linear ? Dict(:alpha => collect(0.1:0.1:1)) : Dict(:maxdepth => collect(1:2:10))
                    d = create_hp_dict(model)
                    d[:model] = [model]

                    df = deepcopy(X_missing)
                    df[!,:Test] = test_ind

                    start = time()
                    (opt_imp_then_reg, μ), bestparams = PHD.regress_cv(Y, df; model=:joint, parameter_dict=d)
                    δt = (time() - start)
                    
                    # plist = model == :linear ? [Dict(:alpha => i) for i in collect(0.1:0.1:1)] : 
                    #     (model == :tree ? [Dict(:maxdepth => i) for i in collect(1:2:10)] : 
                    #     # (model == :rf ? [Dict(:maxdepth => i) for i in collect(1:2:10)] : 
                    #     [Dict(:hidden_nodes => i) for i in collect(5:5:20)]) #)
                    # d = Dict(:modeltype => [model], :parameter_dict => plist)

                    # df = deepcopy(X_missing)
                    # df[!,:Test] = test_ind

                    # start = time()
                    # (opt_imp_then_reg, μ), bestparams = PHD.regress_cv(Y, df; model=:joint, parameter_dict=d)
                    # # opt_imp_then_reg, bestparams, μ = PHD.impute_then_regress_cv(Y, df; modeltype=model, parameter_dict=d)
                    # # (opt_imp_then_reg, μ), bestparams, μ = PHD.impute_then_regress_cv(Y, df; modeltype=model, parameter_dict=d)
                    # δt = (time() - start)

                    R2, OSR2 = PHD.evaluate(Y, PHD.mean_impute(df, μ), opt_imp_then_reg)
                    R2l, OSR2l = PHD.stratified_evaluate(Y, PHD.mean_impute(df, μ), opt_imp_then_reg, patidx, subsetpattern=subsetpattern)   
                    push!(results_table, [dname, SNR, k, missingness_proba, iter, string("Joint Imp-then-Reg - ", model), R2, OSR2, R2l, OSR2l,
                                Matrix(μ[:,canbemissing])[1,:], 
                                δt, model == :linear ? bestparams[:parameter_dict][:alpha] : bestparams[:parameter_dict][:maxdepth]])
                    CSV.write(savedir*filename, results_table)
                end
            end

        end
    end 
# end