###################################
### augment.jl
### Functions to add PHD features
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Create the matrix Z such that Z_ij = 1 if X_ij is missing and 0 otherwise
"""
function indicatemissing(df::DataFrame)
	result = DataFrame()
	for name in names(df)
		if !(name in [:Y, :Test])
			result[Symbol("$(name)_missing")] = Int.(ismissing.(df[name]))
		end
	end
	return result
end

"""
	Create the outer product between the data matrix and the missingness indicators
"""
function augmentaffine(df::DataFrame)
	newdf = zeroimpute(df)
	Z = indicatemissing(df::DataFrame)
	result = hcat(newdf, Z)
	for missingname in names(Z)
		for name in setdiff(names(newdf), [:Test, :Y])
			result[Symbol("$(name)_$missingname")] = newdf[name] .* Z[missingname]
		end
	end
	return result
end