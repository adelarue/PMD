###################################
### creation/rdatasets.jl
### Missing data datasets from RDatasets.jl
### Authors: XXXX
### Adapted from UCIData.jl
###################################

"Make ID column"
function makeid(df::DataFrame, i::Int, id_indices)
	if isempty(id_indices)
		id = "id_$i"
	else
		id = join([df[i, j] for j in id_indices], "_")
	end
end

"Process categorical/continuous variables and target"
function process_dataset(df::DataFrame;
					     target_index::Int=-1,
					     feature_indices=1:0,
					     id_indices=1:0,
					     categoric_indices=1:0)
	output_names = Symbol[]
	output_df = DataFrame()

	output_df.id = [makeid(df, i, id_indices) for i in 1:nrow(df)]
	push!(output_names, :id)

 	# Construct output values
	categoric_indices_set = Set(categoric_indices)
	for i in feature_indices
		if i in categoric_indices_set
			push!(output_names, :C)
		else
			push!(output_names, :N)
		end
		output_df = hcat(output_df, df[!, i], makeunique=true)
	end
	names!(output_df, output_names, makeunique=true)

	if target_index > 0
		output_df.target = df[!, target_index]
		output_df = filter(row -> !ismissing(row[:target]), output_df)
	end
	return output_df
end

"Find RDatasets with more than k missing values"
function rdatasets_missing(k::Int = 5)
	missinglist = []
	data = RDatasets.datasets()
	for i = 1:nrow(data)
		df = RDatasets.dataset(data[i, :Package], data[i, :Dataset])
		if sum(ismissing.(df |> Matrix)) > k
			push!(missinglist, (data[i, :Package], data[i, :Dataset]))
		end
    end
    return missinglist
end

"""
	Given an RDataset with missing values, put it in the UCIData format
"""
function format_dataset(package::AbstractString, dataset::AbstractString)
	df = RDatasets.dataset(package, dataset)
	if package == "COUNT"
		if dataset == "loomis"
			return process_dataset(df, target_index = 1,
			                       feature_indices = [2, 3, 8],
			                       categoric_indices = [2, 3, 8])
		elseif dataset == "ships"
			return process_dataset(df, target_index = 1,
			                       feature_indices = 2:7,
			                       categoric_indices = setdiff(2:7, 6))
		end
	elseif package == "Ecdat"
		if dataset == "Accident"
			return nothing # this is the same as ships
		elseif dataset == "BudgetFood"
			return process_dataset(df, target_index = 1,
			                       feature_indices = 2:6,
			                       categoric_indices = [6])
		elseif dataset == "Garch"
			return nothing # only one missing value due to time lag
		elseif dataset == "Hdma"
			return process_dataset(df, target_index = 13,
			                       feature_indices = 1:12,
			                       categoric_indices = [6, 7, 8, 9, 11, 12])
		elseif dataset == "MCAS"
			return process_dataset(df, target_index = 15,
			                       feature_indices = setdiff(4:17, 15))
		elseif dataset == "Males"
			return process_dataset(df, target_index = 9,
			                       feature_indices = setdiff(2:12, 9),
			                       categoric_indices = setdiff(2:12, [2, 3, 4]))
		elseif dataset == "Mofa"
			return process_dataset(df, target_index = 5,
			                       feature_indices = 1:4,
			                       categoric_indices = [1])
		elseif dataset == "PSID"
			return process_dataset(df, target_index = 5,
			                       feature_indices = setdiff(2:8, 5),
			                       categoric_indices = [8])
		elseif dataset == "RetSchool"
			allowmissing!(df)
			df[df[!, :NoDadEd] .== 1, :DadEd] .= missing
			df[df[!, :NoMomEd] .== 1, :MomEd] .= missing
			return process_dataset(df, target_index = 1,
			                       feature_indices = setdiff(2:17, [11, 12]),
			                       categoric_indices = [4, 5, 6, 7, 8, 9, 10, 15, 17])
		elseif dataset == "Schooling"
			allowmissing!(df)
			df[df[!, :NoDadEd] .== 1, :DadEd] .= missing
			df[df[!, :NoMomEd] .== 1, :MomEd] .= missing
			return process_dataset(df, target_index = 19,
			                       feature_indices = setdiff(1:28, [11, 13, 19, 22]),
			                       categoric_indices = [1, 2, 3, 4, 5, 6, 14, 15, 16, 17,
			                       						18, 20, 21, 23, 26, 27])
		end
	elseif package == "HSAUR"
		if dataset == "BtheB"
			return nothing # longitudinal missingness
		elseif dataset == "Forbes2000"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 3:8,
			                       categoric_indices = [3, 4])
		elseif dataset == "schizophrenia2"
			return nothing # longitudinal
		end
	elseif package == "HistData"
		if dataset in ["Cavendish", "Fingerprints", "OldMaps"]
			return nothing # nothing to predict
		end
	elseif package == "ISLR"
		if dataset == "Hitters"
			return process_dataset(df, target_index = 19,
			                       feature_indices = setdiff(1:20, 19),
			                       categoric_indices = [14, 15, 20])
		end
	elseif package == "KMsurv"
		if dataset == "bcdeter"
			return nothing
		end
	elseif package == "MASS"
		if dataset == "Cars93"
			for i = 1:nrow(df), j in names(df)
				allowmissing!(df)
				if !ismissing(df[i, j]) && df[i, j] == "rotary"
					df[i, j] = missing
				end
			end
			return process_dataset(df, target_index = 5,
			                       feature_indices = union([1, 3], 7:26),
			                       categoric_indices = union([1, 3, 9, 10, 16, 26]))
		elseif dataset == "Pima.tr2"
			return process_dataset(df, target_index = 8,
			                       feature_indices = 1:7)
		elseif dataset == "biopsy"
			return nothing # already in UCI data
		elseif dataset == "survey"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 1:12,
			                       categoric_indices = [1, 4, 5, 7, 8, 9, 11])
		end
	elseif package == "Zelig"
		if dataset == "coalition2"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 1:8,
			                       categoric_indices = [2, 3, 6, 8])
		elseif dataset in ["free1", "free2", "immigration"]
			return nothing # weird datasets, potentially already imputed
		end
	elseif package == "adehabitatLT"
		return nothing # weird datasets, not many missing values
	elseif package == "boot"
		if dataset == "neuro"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 1:6)
		end
	elseif package == "car"
		if dataset == "Chile"
			return process_dataset(df, target_index = 8,
			                       feature_indices = 1:7,
			                       categoric_indices = [1, 3, 5])
		elseif dataset == "Davis"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 1:5,
			                       categoric_indices = [1])
		elseif dataset == "Freedman"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 2:5)
		elseif dataset == "Hartnagel"
			# remove data from 1950 because it was interpolated
			df = filter(row -> row[:Year] != 1950, df)
			return process_dataset(df, target_index = -1,
			                       feature_indices = 1:8)
		elseif dataset == "SLID"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 1:5,
			                       categoric_indices = 4:5)
		elseif dataset == "UN"
			return nothing  # too few columns
		end
	elseif package == "cluster"
		if dataset == "plantTraits"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 2:32)
		elseif dataset == "votes.repub"
			return nothing # missingness patterns linked to existence of states
		end
	elseif package == "datasets"
		if dataset == "airquality"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 1:4)
		elseif dataset == "attenu"
			return process_dataset(df, target_index = 5,
			                       feature_indices = [1, 2, 4],
			                       categoric_indices = [1])
		end
	elseif package == "gap"
		if dataset == "PD"
			return nothing # weird medical dataset
		elseif dataset == "mao"
			return nothing # weird medical dataset
		elseif dataset == "mhtdata"
			return nothing # weird medical dataset
		end
	elseif package == "ggplot2"
		if dataset == "movies"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 2:24,
			                       categoric_indices = [17])
		elseif dataset == "msleep"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 3:11,
			                       categoric_indices = 3:5)
		end
	elseif package == "mlmRev"
		if dataset == "Gcsemv"
			return process_dataset(df, target_index = -1,
			                       feature_indices = union([1], 3:5),
			                       categoric_indices = [1, 3])
		elseif dataset == "star"
			return process_dataset(df, target_index = 10,
			                       feature_indices = union(2:8, 11:14, 16:17),
			                       categoric_indices = union(2:6, [8], 11:14))
		end
	elseif package == "plyr"
		if dataset == "baseball"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 3:23,
			                       categoric_indices = [3, 5, 6])
		end
	elseif package == "pscl"
		if dataset == "politicalInformation"
			return process_dataset(df, target_index = 1,
			                       feature_indices = 2:7,
			                       categoric_indices = [2, 3, 5, 6, 7])
		end
	elseif package == "psych"
		return nothing  # datasets have no description or very few missing values
	elseif package == "reshape2"
		if dataset == "french_fries"
			return process_dataset(df, target_index = -1,
			                       feature_indices = union(2:3, 6:10))
		end
	elseif package == "robustbase"
		if dataset == "airmay"
			return nothing  # subset of air quality dataset
		elseif dataset == "ambientNOxCH"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 2:14)
		end
	elseif package == "rpart"
		if dataset == "car.test.frame"
			return process_dataset(df, target_index = 2,
			                       feature_indices = 3:9,
			                       categoric_indices = [3, 6])
		elseif dataset == "stagec"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 1:8,
			                       categoric_indices = [2, 4, 8])
		end
	elseif package == "sem"
		if dataset == "Tests"
			return process_dataset(df, target_index = -1,
			                       feature_indices = 1:6)
		end
	elseif package == "survival"
		if dataset == "cancer"
			return process_dataset(df, target_index = 2,
			                       feature_indices = 3:10,
			                       categoric_indices = [3, 5])
		elseif dataset == "colon"
			return nothing  # low missingness, weird time series component
		elseif dataset == "lung"
			return nothing  # duplicate of cancer
		elseif dataset == "mgus"
			return process_dataset(df, target_index = 8,
			                       feature_indices = union(2:7, 9:12),
			                       categoric_indices = [3, 5])
		elseif dataset == "pbc"
			return process_dataset(df, target_index = 3,
			                       feature_indices = union(4:20),
			                       categoric_indices = [4, 6, 10, 20])
		end
	end
end

"""
	Create datasets from RDatasets repository
"""
function create_r_datasets()
	Random.seed!(1515)

	folderlist = readdir("$(@__DIR__)/../../datasets/")

	# Get dataset, and impute it using missForest

	for (p, d) in rdatasets_missing(5)
		n = p * "-" * d
        if n ∉ folderlist || "X_missing.csv" ∉ readdir("$(@__DIR__)/../../datasets/"*n*"/")
            @show n
    		df = format_dataset(p, d)
    		if df !== nothing
				df = onehotencode(df) #One-hot encode categorical columns
	    		if any(.!completecases(df)) || any([endswith(k,"_Missing") for k in String.(names(df))])
	                select!(df, Not(intersect(Symbol.(names(df)), [:id])))
	    			if ncol(df) > 1
	    				impute_data(df, n)
	    			end
	    		end
	    	end
        end
	end
end
