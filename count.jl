###################################
### count.jl
### Functions to count number of missing values
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

using DataFrames, CSV

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

function count_missing_columns(datasetname::AbstractString)
	df = CSV.read("../datasets/$datasetname/1/X_missing.csv")
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

function missing_percentage(datasetname::AbstractString)
	df = CSV.read("../datasets/$datasetname/1/X_missing.csv")
	return missing_percentage(df)
end

"""
	Given a dataset name, return the dataset size
"""
function size(datasetname::AbstractString) #Note from Jean: My install of julia gets confused of this renaming of size.
	df = CSV.read("../datasets/$datasetname/1/X_missing.csv")
	return Base.size(df)
end
