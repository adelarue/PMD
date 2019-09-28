###################################
### create_sleep_datasets.jl
### Script to create regression datasets with missing values
### Uses the sleep dataset from the R VIM package,
### 	see https://cran.r-project.org/web/packages/VIM/VIM.pdf
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

using RDatasets, RCall, DataFrames, CSV
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

Random.seed!(1515)
NUM_IMPUTATIONS = 20
NOISE = 0.2
TEST_FRACTION = 0.3

# Get dataset, and impute it a bunch of times using CART
R"library(VIM)"
R"data(sleep, package='VIM')"
R"sleep = as.data.frame(scale(sleep))"
R"library(mice)"
R"imputed = mice(sleep, m = $NUM_IMPUTATIONS, method='cart')"
@rget sleep
nan_to_missing!(sleep)
# Train-test split
num_test_points = Int(floor(nrow(sleep) * TEST_FRACTION))
test_points = shuffle(vcat(zeros(Int, nrow(sleep) - num_test_points), ones(Int, num_test_points)))

# Get dependent variable, then hide the missing values once again
for i = 1:NUM_IMPUTATIONS
	# Get the i-th imputed dataset from mice + cart
    R"imputedsleep = complete(imputed, action = $i)"
    @rget imputedsleep
    # compute the dependent variable
    sleep[:Y] = imputedsleep[:Sleep] .+ randn(nrow(imputedsleep)) * NOISE
    sleep[:Test] = test_points
    # Save the dataset
    filename = @sprintf("datasets/sleep-%.2f-%02d.csv", NOISE, i)
    CSV.write(filename, sleep)
end
