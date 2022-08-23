###################################
### evaluate.jl
### Evaluate the quality of a predictive model
### Authors: Arthur Delarue, Jean Pauphilet, 2022
###################################

"""
	Evaluate the fit quality of a model on a dataset. Returns in- and out-of-sample
    Train/Test identified with the :Test column
"""
function evaluate(Y::Vector, df::DataFrame, model::Union{DataFrame,DecisionTree.Root,DecisionTree.Ensemble,Chain,GreedyModel,Tuple{Any, DataFrame}})
	prediction = predict(df, model)
	# @show (prediction)
	testavail = "Test" ∈ names(df)
    if !testavail #If no test column, all dataset is considered training
        df[!,:Test] .= 0
    end

	trainmean = Statistics.mean(Y[df[:,:Test] .== 0])
	R2 = rsquare(Y[df[:,:Test] .== 0], prediction[df[:,:Test] .== 0]; baseline=trainmean)
	if !testavail #If no test column, all dataset is considered training
        return R2, NaN
	else 
		OSR2 = rsquare(Y[df[:,:Test] .== 1], prediction[df[:,:Test] .== 1]; baseline=trainmean)
		return R2, OSR2
    end 
end
function evaluate(Y::BitArray{1}, df::DataFrame, model::Union{DataFrame,DecisionTree.Root,DecisionTree.Ensemble,Chain,GreedyModel,Tuple{Any, DataFrame}};
				  metric::AbstractString="auc")
	prediction = predict(df, model)
	# if model[1, :Logistic]
    if metric == "logloss"
        ll = logloss(Y[df[:,:Test] .== 0], prediction[df[:,:Test] .== 0])
        osll = logloss(Y[df[:,:Test] .== 1], prediction[df[:,:Test] .== 1])
        return ll, osll
    elseif metric == "auc"
        return auc(Y[df[:,:Test] .== 0], prediction[df[:,:Test] .== 0]),
                auc(Y[df[:,:Test] .== 1], prediction[df[:,:Test] .== 1])
    else
        error("Unknown metric: $metric (only supports 'logloss', 'auc')")
    end
	# else
	# 	error("Continuous model evaluated on binary vector")
	# end
end

function stratified_evaluate(Y, df::DataFrame, model, patternidx; subsetpattern=unique(patternidx))
    R2list = []; OSR2list = []
    for p in sort(intersect(unique(patternidx), subsetpattern))
#         try 
            R2, OSR2 = evaluate(Y[patternidx .== p], df[patternidx .== p, :], model)
            push!(R2list, R2)
            push!(OSR2list, OSR2)
#         catch 
#             push!(R2list, 0)
#             push!(OSR2list, 0)
#         end
    end
    return R2list, OSR2list
end

function accuracy(Y::Vector, pred; baseline=mean(Y))
	rsquare(Y, pred; baseline)
end
function accuracy(Y::BitArray{1}, pred)
	auc(Y, pred)
end

"Compute Rsquare"
function rsquare(Y::Vector, pred; baseline=mean(Y))
	SST = mean((Y .- baseline) .^ 2)
	R2 = 1 - mean( (Y .- pred).^ 2 )/SST
	# @show SST, R2
	return R2
end

"Compute logistic loss of a particular prediction"
function safelog(x)
	if x < 1e-10
		return log(1e-10)
	elseif x > 1e10 
		return log(1e10)
	else 
		return log(x)
	end
end
function logloss(actual::BitArray{1}, predicted::Vector)
	return sum(actual .* safelog.(predicted) .+ (1 .- actual) .* safelog.(1 .- predicted))
end
logloss(actual::BitArray{1}, constant::Real) = logloss(actual, constant .* ones(length(actual)))

"Compute AUC of predictions"
function auc(actual::BitArray{1}, predicted::Vector)
	@assert length(actual) == length(predicted)
	r = StatsBase.tiedrank(predicted)
	n_pos = sum(actual)
	n_neg = length(actual) - n_pos
	AUC = (sum(r[actual]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
	# return AUC #Raw AUC between 0.5 and 1
	return 2*(AUC -.5) #Adjusted AUC between 0 and 1 --Comparable with R2
end