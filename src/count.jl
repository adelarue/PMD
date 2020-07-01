###################################
### count.jl
### Functions to count number of missing values
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Count number of columns with at least one missing value
"""
function count_missing_columns(df::DataFrame)
	counter = 0
	for name in names(df)
		if any(ismissing.(df[!, name]))
			counter += 1
		end
	end
	return counter
end

"""
	Given the name of a dataset in the datasets folder, count the number of missing
		columns with at least one missing value
"""
function count_missing_columns(datasetname::AbstractString)
	df = CSV.read("$(@__DIR__)/../datasets/$datasetname/1/X_missing.csv")
	return count_missing_columns(df)
end

"""
	Compute total percentage of missing values among columns with missing values
"""
function missing_percentage(df::DataFrame)
	counter = 0
	for i = 1:nrow(df), name in names(df)
		if ismissing(df[i, name])
			counter += 1
		end
	end
	return counter / (nrow(df) * count_missing_columns(df)) * 100.0
end

"""
	Given the name of a dataset in the datasets folder, count the number of missing
		columns with at least one missing value
"""
function missing_percentage(datasetname::AbstractString)
	df = CSV.read("$(@__DIR__)/../datasets/$datasetname/1/X_missing.csv")
	return missing_percentage(df)
end

"""
	Given a dataset name, return the dataset size
"""
function size(datasetname::AbstractString)
	df = CSV.read("$(@__DIR__)/../datasets/$datasetname/1/X_missing.csv")
	return Base.size(df)
end

"""
	Return unique missingness patterns in the provided dataframe
"""
function missing_patterns(df::DataFrame)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test, :Y])
	return unique([ismissing.(convert(Vector, df[i, cols])) for i = 1:nrow(df)])
end
count_missing_patterns(df::DataFrame) = length(missing_patterns(df))

"""
	Return unique missingness patterns ordered by count
"""
function missing_patterns_countmap(df::DataFrame)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test, :Y])
	patterns = [ismissing.(convert(Vector, df[i, cols])) for i = 1:nrow(df)]
	patternmap = StatsBase.countmap(patterns)
	patterns = collect(keys(patternmap))
	counts = Int[]
	for pattern in patterns
		push!(counts, patternmap[pattern])
	end
	p = reverse(sortperm(counts))
	return patterns[p], counts[p]
end
