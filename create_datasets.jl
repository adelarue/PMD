###################################
### create_sleep_datasets.jl
### Script to create regression datasets with missing values
### Uses the sleep dataset from the R VIM package,
### 	see https://cran.r-project.org/web/packages/VIM/VIM.pdf
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

using Pkg
Pkg.activate(".")

using Revise
using PHD
using UCIData, RCall, Random, DataFrames, CSV

"""
	Create datasets from UCI data repository
"""
function create_uci_datasets()
	Random.seed!(1515)

	folderlist = readdir("$(@__DIR__)/datasets/")

	# Get dataset, and impute it using missForest
	PHD.load_R_library("VIM")
	R"data(sleep, package='VIM')"
	R"sleep = as.data.frame(scale(sleep))"
	@rget sleep
	PHD.nan_to_missing!(sleep)
	PHD.impute_data(sleep, "sleep")

	for task in ["classification", "regression"]
		for n in UCIData.list_datasets(task)
	        if n ∉ folderlist || "X_missing.csv" ∉ readdir("$(@__DIR__)/datasets/"*n*"/")
	            @show n
	    		df = UCIData.dataset(n)
    			df = PHD.onehotencode(df) #One-hot encode categorical columns
	    		if any(.!completecases(df)) || any([endswith(k,"_Missing") for k in String.(names(df))])
	                select!(df, Not(intersect(names(df), [:id])))

	    			if ncol(df) > 1
	    				PHD.impute_data(df, n)
	    			end
	    		end
	        end
		end
	end
end

create_uci_datasets()
PHD.create_r_datasets()
