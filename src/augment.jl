###################################
### augment.jl
### Functions to add PHD features
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Remove columns that are uniformly zero
"""
function removezerocolumns(df::DataFrame)
	tokeep = []
	for name in names(df)
		if !all(abs.(df[!, name]) .< 1e-9)
			push!(tokeep, name)
		end
	end
	return df[!, tokeep]
end

"""
	Create the matrix Z such that Z_ij = 1 if X_ij is missing and 0 otherwise
"""
function indicatemissing(df::DataFrame; removezerocols::Bool=false)
	result = DataFrame()
	for name in names(df)
		if !startswith(String(name), "C") && !(name in [:Y, :Test]) #if not categorical nor Test/Y
			result[:,Symbol("$(name)_missing")] = (Int.(ismissing.(df[name])))
		end
	end
	if removezerocols
		result = removezerocolumns(result)
	end
	return result
end

"""
	Create the outer product between the data matrix and the missingness indicators
"""
function augmentaffine(df::DataFrame; removezerocols::Bool = false)
	newdf = zeroimpute(df)
	Z = indicatemissing(df, removezerocols=true)
	result = hcat(newdf, Z)
	for missingname in names(Z)
		for name in setdiff(names(newdf), [:Test, :Y])
			result[!, Symbol("$(name)_$missingname")] = newdf[!, name] .* Z[!, missingname]
		end
	end
	if removezerocols
		result = removezerocolumns(result)
	end
	return result
end