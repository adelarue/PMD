###################################
### regress.jl
### Functions to perform linear regression
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Perform linear regression on a dataframe
	Args:
		- a dataframe with no missing values
	Parameters:
		- lasso: true if we use lasso (penalty chosen using cross-validation),
				 false if we do not regularize
		- alpha: elastic net parameter (1.0 for lasso, 0.0 for ridge)
		- missing_penalty: how much to penalize augmented columns
				 (columns that contain missing in their name). Regular columns have penalty 1.0
	Returns:
		- a dataframe with a single row, with the same column names as the data
		except Test, and an additional Offset containing the constant term
"""
function regress(Y::Array{Float64}, df::DataFrame;
				 lasso::Bool=false, alpha::Real=0.8, missing_penalty::Real=1.0)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	X = convert(Matrix, df[df[!, :Test] .== 0, cols])
	y = convert(Array, Y[df[!, :Test] .== 0])
	coefficients = DataFrame()
	if lasso
		penalty_factor = ones(length(cols))
		for (i, col) in enumerate(cols)
			if occursin("_missing", string(col))
				penalty_factor[i] = missing_penalty
			end
		end
		cv = glmnetcv(X, y, alpha=alpha, penalty_factor=penalty_factor)
		for (i, col) in enumerate(cols)
			coefficients[!,col] = ([cv.path.betas[i, argmin(cv.meanloss)]])
		end
		coefficients[:Offset] = cv.path.a0[argmin(cv.meanloss)]
	else
		path = glmnet(X, y)
		for (i, col) in enumerate(cols)
			coefficients[!,col] = [path.betas[i, end]]
		end
		coefficients[:Offset] = path.a0[end]
	end
	coefficients[:Logistic] = false
	return coefficients
end

function regress(Y::BitArray{1}, df::DataFrame;
				 lasso::Bool=false, alpha::Real=0.8, missing_penalty::Real=1.0)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	X = convert(Matrix, df[df[!, :Test] .== 0, cols])
	y = convert(Array, Y[df[!, :Test] .== 0])

	freq = mean(1.0.*Y)
	w = (1/freq).*Y .+ (1/(1 - freq)).*(1. .- Y); #class weights
	coefficients = DataFrame()
	if lasso
		penalty_factor = ones(length(cols))
		for (i, col) in enumerate(cols)
			if occursin("_missing", string(col))
				penalty_factor[i] = missing_penalty
			end
		end
		cv = glmnetcv(X, hcat(Float64.(.!y), Float64.(y)), GLMNet.Binomial(),
		              alpha=alpha, penalty_factor=penalty_factor, weights=w)
		for (i, col) in enumerate(cols)
			coefficients[!,col] = ([cv.path.betas[i, argmin(cv.meanloss)]])
		end
		coefficients[:Offset] = cv.path.a0[argmin(cv.meanloss)]
	else
		path = glmnet(X, hcat(Float64.(.!y), Float64.(y)), GLMNet.Binomial(), weights=w)
		for (i, col) in enumerate(cols)
			coefficients[!,col] = [path.betas[i, end]]
		end
		coefficients[:Offset] = path.a0[end]
	end
	coefficients[:Logistic] = true
	return coefficients
end

"""
	Predict the dependent variable given a regression model
"""
function predict(df::DataFrame, model::DataFrame)
	prediction = model[1, :Offset] .* ones(nrow(df))
	for name in setdiff(Symbol.(names(model)), [:Offset, :Logistic])
		prediction .+= (model[1, name]*df[:,name])
	end
	if model[1, :Logistic]
		return sigmoid.(prediction)
	else
		return prediction
	end
end

"""
	Evaluate the fit quality of a linear model on a dataset
"""
function evaluate(Y::Vector, df::DataFrame, model::DataFrame)
	prediction = predict(df, model)
	if model[1, :Logistic]
		error("Logistic model evaluated on continuous vector")
	else
		trainmean = Statistics.mean(Y[df[:,:Test] .== 0])
		SST = sum((Y[df[:,:Test] .== 0] .- trainmean) .^ 2)
		OSSST = sum((Y[df[:,:Test] .== 1] .- trainmean) .^ 2)

		R2 = 1 - sum((Y[df[:,:Test] .== 0] .- prediction[df[:,:Test] .== 0]) .^ 2)/SST
		OSR2 = 1 - sum((Y[df[:,:Test] .== 1] .- prediction[df[:,:Test] .== 1]) .^ 2)/OSSST
		return R2, OSR2
	end
end
function evaluate(Y::BitArray{1}, df::DataFrame, model::DataFrame;
				  metric::AbstractString="logloss")
	prediction = predict(df, model)
	if model[1, :Logistic]
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
	else
		error("Continuous model evaluated on binary vector")
	end
end

"Compute logistic loss of a particular prediction"
function logloss(actual::BitArray{1}, predicted::Vector)
	return sum(actual .* log.(predicted) .- (1 .- actual) .* log.(1 .- predicted))
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
