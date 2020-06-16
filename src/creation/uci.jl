###################################
### creation/uci.jl
### Missing data datasets from UCI repository
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Create datasets from UCI data repository
"""
function create_uci_datasets()
	Random.seed!(1515)

	folderlist = readdir("$(@__DIR__)/../../datasets/")

	# Get dataset, and impute it using missForest
	load_R_library("VIM")
	R"data(sleep, package='VIM')"
	R"sleep = as.data.frame(scale(sleep))"
	@rget sleep
	nan_to_missing!(sleep)
	impute_data(sleep, "sleep")

	for task in ["classification", "regression"]
		for n in UCIData.list_datasets(task)
	        if n ∉ folderlist || "X_missing.csv" ∉ readdir("$(@__DIR__)/../../datasets/"*n*"/")
	            @show n
	    		df = UCIData.dataset(n)
    			df = onehotencode(df) #One-hot encode categorical columns
	    		if any(.!completecases(df)) || any([endswith(k,"_Missing") for k in String.(names(df))])
	                select!(df, Not(intersect(names(df), [:id])))

	    			if ncol(df) > 1
	    				impute_data(df, n)
	    			end
	    		end
	        end
		end
	end
end
