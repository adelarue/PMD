###################################
### regress.jl
### Functions to perform linear regression
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

using DataFrames, GLMNet
using Statistics

"""
	Perform linear regression on a dataframe
	Args:
		- a dataframe with no missing values
	Parameters:
		- lasso: true if we use lasso (penalty chosen using cross-validation),
				 false if we do not regularize
	Returns:
		- a dataframe with a single row, with the same column names as the data
		except Y and Test, and an additional Offset containing the constant term
"""
function regress(df::DataFrame; lasso::Bool=false)
	cols = setdiff(names(df), [:Test, :Y])
	X = convert(Matrix, df[df[:Test] .== 0, cols])
	y = convert(Array, df[df[:Test] .== 0, :Y])
	coefficients = DataFrame()
	if lasso
		cv = glmnetcv(X, y)
		for (i, col) in enumerate(cols)
			coefficients[col] = [cv.path.betas[i, argmin(cv.meanloss)]]
		end
		coefficients[:Offset] = cv.path.a0[argmin(cv.meanloss)]
	else
		path = glmnet(X, y)
		for (i, col) in enumerate(cols)
			coefficients[col] = [path.betas[i, end]]
		end
		coefficients[:Offset] = path.a0[end]
	end
	return coefficients
end

"""
	Predict the dependent variable given a regression model
"""
function predict(df::DataFrame, model::DataFrame)
	prediction = model[1, :Offset] * ones(nrow(df))
	for name in setdiff(names(model), [:Offset])
		prediction .+= df[name] .* model[1, name]
	end
	return prediction
end

"""
	Evaluate the fit quality of a linear model on a dataset
"""
function evaluate(df::DataFrame, model::DataFrame)
	trainmean = Statistics.mean(df[df[:Test] .== 0, :Y])
	SST = sum((df[df[:Test] .== 0, :Y] .- trainmean) .^ 2)
	OSSST = sum((df[df[:Test] .== 1, :Y] .- trainmean) .^ 2)
	prediction = predict(df, model)
	R2 = 1 - sum((df[df[:Test] .== 0, :Y] - prediction[df[:Test] .== 0]) .^ 2)/SST
	OSR2 = 1 - sum((df[df[:Test] .== 1, :Y] - prediction[df[:Test] .== 1]) .^ 2)/OSSST
	return R2, OSR2
end
