###################################
### impute.jl
### Functions to impute missing values before performing regression
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

using RCall, DataFrames, CSV

"""
	Impute missing values in a dataset using mice
	Imputes training set alone, then testing set with training set
"""
function mice(df::DataFrame)
	result = deepcopy(df)
	@rput df
	R"library(mice)"
	R"library(dplyr)"
	R"train = select(df, -c('Id'))"

	R"colnames <- names(train)"
	R"names(train) <- make.names(colnames, unique=TRUE)"
	R"imputed = mice(as.data.frame(train), m=2, printFlag=F)"
	R"imputedtrain = complete(imputed, action=1)"
	R"names(imputedtrain) <- colnames"
	@rget imputedtrain
	imputedtrain[!,:Id] = df[:,:Id]
	# R"trainplus = select(as.data.frame(df), -c(Test))"
	# # for some reason, sometimes columns are lists
	# R"trainplus = t(apply(trainplus, 1, unlist))"
	# R"imputedplus = mice(as.data.frame(trainplus), m=1, printFlag=F)"
	# R"imputedtest = select(subset(mutate(complete(imputedplus, action=1), Test=df$Test), Test==1), -Test)"
	# @rget imputedtest
	# result[result.Test .== 0, setdiff(names(result), [:Test])] .= imputedtrain
	# result[result.Test .== 1, setdiff(names(result), [:Test])] .= imputedtest
	return imputedtrain
end

"""
	Impute all missing values as mean
"""
function compute_mean(df::DataFrame)
	numcols = filter(t->startswith(string(t),"N"), names(df))
	nummeans = []
	for c in numcols
	    push!(nummeans, mean(skipmissing(df[:,c])))
	end
	return DataFrame(nummeans', numcols)
end
function mean_impute(df::DataFrame, means)
	result = deepcopy(df)
	for n in names(means)
		result[!,n] = convert(Array{Union{Missing,Float64},1},result[:,n])
		for i=1:nrow(df)
			if ismissing(result[i, n])
				result[i,n] = means[1,n]
			end
		end
	end
	return result
end

"""
	Impute all missing values as zeros
"""
function zeroimpute(df::DataFrame)
	result = deepcopy(df)
	for i=1:nrow(df), name in names(df)
		if ismissing(result[i, name])
			result[i, name] = 0
		end
	end
	return result
end
