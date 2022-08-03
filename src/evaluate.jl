###################################
### evaluate.jl
### Evaluate the quality of a predictive model
### Authors: Arthur Delarue, Jean Pauphilet, 2022
###################################

"""
	Evaluate the fit quality of a model on a dataset. Returns in- and out-of-sample
    Train/Test identified with the :Test column
"""
function evaluate(Y::Vector, df::DataFrame, model::Union{DataFrame,DecisionTree.Node,Chain})
	prediction = predict(df, model)
    if "Test" ∉ names(df) #If no test column, all dataset is considered training
        error("Missing Test column in the data (second argument)")
    end
	# if model[1, :Logistic]
	# 	error("Logistic model evaluated on continuous vector")
	# else
    trainmean = Statistics.mean(Y[df[:,:Test] .== 0])
    SST = sum((Y[df[:,:Test] .== 0] .- trainmean) .^ 2)
    OSSST = sum((Y[df[:,:Test] .== 1] .- trainmean) .^ 2)

    R2 = 1 - sum((Y[df[:,:Test] .== 0] .- prediction[df[:,:Test] .== 0]) .^ 2)/SST
    OSR2 = 1 - sum((Y[df[:,:Test] .== 1] .- prediction[df[:,:Test] .== 1]) .^ 2)/OSSST
    return R2, OSR2
end
function evaluate(Y::BitArray{1}, df::DataFrame, model::Union{DataFrame,DecisionTree.Node,Chain};
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
	return AUC
end