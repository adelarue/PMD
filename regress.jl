###################################
### regress.jl
### Functions/script to perform linear regression
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

using DataFrames, GLMNet
using Statistics

# To do: add regularization capabilities
"""
	Perform linear regression on a dataframe
	Assumes that the dataframe has no missing values
	Returns a dataframe with a single row, with the same column names as the data
		except Y and Test, and an additional Offset containing the constant term
"""
function regress(df::DataFrame)
	cols = setdiff(names(df), [:Test, :Y])
	X = convert(Array, df[df[:Test] .== 0, cols])
	y = convert(Array, df[df[:Test] .== 0, :Y])
	model = glmnet(X, y)
	coefficients = DataFrame()
	for (i, col) in enumerate(cols)
		coefficients[col] = [model.betas[i, end]]
	end
	coefficients[:Offset] = model.a0[end]
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
