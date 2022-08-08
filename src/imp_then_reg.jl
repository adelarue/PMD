###################################
### imp-then-reg.jl
### Functions to perform joint imputation-then-regression
### Authors: Arthur Delarue, Jean Pauphilet, 2022
###################################

function impute_then_regress_cv(Y::Union{Vector{Float64},BitArray}, data::DataFrame; 
    modeltype::Symbol, parameter_dict::Dict=Dict(), 
    val_fraction::Real=0.2,
    maxiter::Int=10, maxμiter::Int=100, ϵ::Float64=1.)


    X = select(data, Not([:Id]))

    canbemissing = findall([any(ismissing.(X[:,j])) for j in names(X)]) #indicator of missing features
    μ = [mean(X[.!ismissing.(X[:,j]), j]) for j in names(X)] #original mean vect
    σ = [std(X[.!ismissing.(X[:,j]), j]) / sqrt(sum(.!ismissing.(X[:,j]))) for j in names(X)] #SE on the mean

    
    X_imp = mu_impute(X, μ, missing_columns=canbemissing) #First μ-imputed data set

    for epoch in 1:maxiter
        #Step 1: Update prediction model
        model, = regress_cv(Y, X_imp, 
            model = modeltype, parameter_dict = parameter_dict,
            val_fraction=val_fraction) 
        
        initialR2, = evaluate(Y, X_imp, model)
        # println("In-sample R2: ", round(initialR2, digits=4))

        #Step 2: Update μ Step
        bestR2 = initialR2
        δ = zeros(length(μ))
        for _ in 1:maxμiter
            nchanges = 0
            for j in shuffle(canbemissing)    
                bestUpdate = 0
                δ[j] = 1
                for s in -1:2:1
                    newXimp = mu_impute(X, μ + s*ϵ*(σ.*δ), missing_columns=canbemissing)
                    newR2, = evaluate(Y, newXimp, model)
                    if newR2 > bestR2
                        bestR2 = newR2; bestUpdate = s
                    end
                end
                nchanges +=  abs(bestUpdate) #Count the number of coordinates of μ updated
                μ .+= bestUpdate*ϵ*(σ.*δ)        
                δ[j] = 0
            end
            if nchanges == 0
                # @show μepoch
                break
            end
        end
        
        X_imp = mu_impute(X, μ, missing_columns=canbemissing)
        newR2, = evaluate(Y, X_imp, model)  
        # println("In-sample R2 after μ update: ", round(newR2, digits=4))
        # @show (newR2-initialR2)/initialR2
        if epoch > 10 && (newR2-initialR2)/initialR2 < 0.0001
            break
        end
    end

    model, bestparams = regress_cv(Y, X_imp,
        model = modeltype, parameter_dict = parameter_dict,
        val_fraction=val_fraction)

    return model, bestparams, meanvector_to_df(μ, names(X))
end