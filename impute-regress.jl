###################################
### impute-regress.jl
### Script to impute missing values before performing regression
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
	R"imputed = mice(as.data.frame(train), m=1)"
	R"imputedtrain = complete(imputed, action=1)"
	@rget imputedtrain
	R"trainplus = select(as.data.frame(df), -c(Test, Y))"
	# for some reason, sometimes columns are lists
	R"trainplus = t(apply(trainplus, 1, unlist))"
	R"imputedplus = mice(as.data.frame(trainplus), m=1)"
	R"imputedtest = select(subset(mutate(complete(imputedplus, action=1), Test=df$Test), Test==1), -Test)"
	@rget imputedtest
	result[result.Test .== 0, setdiff(names(result), [:Test, :Y])] = imputedtrain
	result[result.Test .== 1, setdiff(names(result), [:Test, :Y])] = imputedtest
	return result
end

data = CSV.read("datasets/sleep-0.2-1.csv")
println(first(data, 5))
println(first(mice(data), 5))
