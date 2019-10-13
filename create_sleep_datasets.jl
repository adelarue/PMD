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

function onehotencode!(df)
    categorical_cols = [k for k in names(df) if startswith(String(k),"C")]

    long_cat = DataFrame(id=[], variable=[], value=[])
    for c in categorical_cols
        for i in 1:size(df,1)
            if !ismissing(df[i,c])
                push!(long_cat, [df[i,:id], string(String(c),"_",df[i,c]), 1])
            else
                push!(long_cat, [df[i,:id], string(String(c),"_","Missing"), 1])
            end
        end
    end

    wide_cat = unstack(long_cat, :id, :variable, :value)
    coalesce.(wide_cat,0)

    select!(df, Not(categorical_cols))
    df = join(df, wide_cat, on=:id)
end

function create_data(df, df_name; NUM_IMPUTATIONS = 10, TEST_FRACTION=.3)
	if !isdir("datasets/"*df_name*"/")
		mkdir("datasets/"*df_name*"/")
	end
	R"library(mice)"
	R"imputed = mice($df, m = $NUM_IMPUTATIONS, method='cart')"
		# @rget sleep
	# nan_to_missing!(df)
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

# Get dataset, and impute it a bunch of times using CART
R"library(VIM)"
R"data(sleep, package='VIM')"
R"sleep = as.data.frame(scale(sleep))"

@rget sleep
# nan_to_missing!(sleep)

create_data(sleep, "sleep"; NUM_IMPUTATIONS = NUM_IMPUTATIONS, TEST_FRACTION = TEST_FRACTION)

for task in ["classification", "regression"]
	for n in UCIData.list_datasets(task)
		@show n
		df = UCIData.dataset(n)
		if any([startswith(k,"C") for k in String.(names(df))])
			onehotencode!(df)
		end
		if any(.!completecases(df)) || any([endswith(k,"_Missing") for k in String.(names(df))])
			# @show n
			for k in [:id, :target]
				if k ∈ names(df)
					deletecols!(df, k)
				end
			end
			if size(df, 2) > 1
				create_data(df, n; NUM_IMPUTATIONS = NUM_IMPUTATIONS, TEST_FRACTION = TEST_FRACTION)
			end
		end
	end
end
