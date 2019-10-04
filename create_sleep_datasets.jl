###################################
### create_sleep_datasets.jl
### Script to create regression datasets with missing values
### Uses the sleep dataset from the R VIM package,
### 	see https://cran.r-project.org/web/packages/VIM/VIM.pdf
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

using RDatasets, RCall, DataFrames, CSV, UCIData
using Random, Printf

function nan_to_missing!(df::DataFrame)
	for name in names(df)
		df[name] = convert(Vector{Any}, df[name])
	end
	for i = 1:nrow(df), name in names(df)
		if isnan(df[i, name])
			df[i, name] = missing
		end
	end
end

function create_data(df, df_name; NUM_IMPUTATIONS = 10, TEST_FRACTION=.3)
	if !isdir("datasets/"*df_name*"/")
		mkdir("datasets/"*df_name*"/")
	end
	R"library(mice)"
	R"imputed = mice($df, m = $NUM_IMPUTATIONS, method='cart')"
		# @rget sleep
	nan_to_missing!(df)
	# Train-test split
	num_test_points = floor(Int, nrow(df) * TEST_FRACTION)
	test_points = shuffle(vcat(zeros(Int, nrow(df) - num_test_points), ones(Int, num_test_points)))

	# Get dependent variable, then hide the missing values once again
	for i = 1:NUM_IMPUTATIONS
		if !isdir("datasets/"*df_name*"/$i/")
			mkdir("datasets/"*df_name*"/$i/")
		end

		# Get the i-th imputed dataset from mice + cart
	    R"imputeddf= complete(imputed, action = $i)"
	    @rget imputeddf
	    # compute the dependent variable
	    df[:Test] = test_points
		imputeddf[:Test] = test_points
	    # Save the dataset
		path = "datasets/"*df_name*"/$i/"
	    CSV.write(path*"X_missing.csv", df)
		CSV.write(path*"X_full.csv", imputeddf)
	end
end

Random.seed!(1515)

NUM_IMPUTATIONS = 20
TEST_FRACTION = 0.3

# # Get dataset, and impute it a bunch of times using CART
# R"library(VIM)"
# R"data(sleep, package='VIM')"
# R"sleep = as.data.frame(scale(sleep))"
#
# @rget sleep
#
# create_data(sleep, "sleep"; NUM_IMPUTATIONS = NUM_IMPUTATIONS, TEST_FRACTION = TEST_FRACTION)

for n in UCIData.list_datasets("classification")
	df = UCIData.dataset(n)
	if any(.!completecases(df))
		for k in [:id, :target]
			if k ∈ names(df)
				deletecols!(df, k)
			end
		end
		create_data(df, n; NUM_IMPUTATIONS = NUM_IMPUTATIONS, TEST_FRACTION = TEST_FRACTION)
	end
end
