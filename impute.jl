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
	R"train = select(subset(as.data.frame(df), Test==0), -c(Test, Y))"
	# for some reason, sometimes columns are lists
	R"train = t(apply(train, 1, unlist))"
	R"imputed = mice(as.data.frame(train), m=1, printFlag=F)"
	R"imputedtrain = complete(imputed, action=1)"
	@rget imputedtrain
	R"trainplus = select(as.data.frame(df), -c(Test, Y))"
	# for some reason, sometimes columns are lists
	R"trainplus = t(apply(trainplus, 1, unlist))"
	R"imputedplus = mice(as.data.frame(trainplus), m=1, printFlag=F)"
	R"imputedtest = select(subset(mutate(complete(imputedplus, action=1), Test=df$Test), Test==1), -Test)"
	@rget imputedtest
	result[result.Test .== 0, setdiff(names(result), [:Test, :Y])] = imputedtrain
	result[result.Test .== 1, setdiff(names(result), [:Test, :Y])] = imputedtest
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
