###################################
### create_sleep_datasets.jl
### Script to create regression datasets with missing values
### Uses the sleep dataset from the R VIM package,
### 	see https://cran.r-project.org/web/packages/VIM/VIM.pdf
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

using RDatasets, RCall, DataFrames, CSV
using Random

Random.seed!(1515)
NUM_IMPUTATIONS = 20
NOISE = 0.2

# Get dataset
R"library(VIM)"
R"data(sleep, package='VIM')"
R"sleep = as.data.frame(scale(sleep))"
R"library(mice)"
R"imputed = mice(sleep, m = $NUM_IMPUTATIONS, method='cart')"
@rget sleep

# Missing values are represented as NAs, change to missing
data = DataFrame()
for name in names(sleep)
    data[name] = []
end
for i = 1:nrow(sleep), name in names(sleep)
    push!(data[name], ifelse(isnan(sleep[i, name]), missing, sleep[i, name]))
end

# Get dependent variable
for i = 1:NUM_IMPUTATIONS
    R"imputedsleep = complete(imputed, action = $i)"
    @rget imputedsleep
    data[:Y] = imputedsleep[:Sleep] .+ randn(nrow(imputedsleep)) * NOISE
    CSV.write("datasets/sleep-$i.csv", data)
end

# Impute missing values using mice with create

