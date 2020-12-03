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
	for name in Symbol.(names(df))
		# if name ∈ [:Id, :Test, :Y] || (!all(abs.(df[!,name]) .< 1e-9))
		if name ∈ [:Id, :Test, :Y] || (var(df[:,name]) > 1e-9) #remove constant value columns
			push!(tokeep, name)
		end
	end
	return df[!, tokeep]
end

"""
	Create the matrix M such that M_ij = 1 if X_ij is missing and 0 otherwise
"""
function indicatemissing(df::DataFrame; removezerocols::Bool=false)
	result = DataFrame()
	for name in Symbol.(names(df))
		if !startswith(String(name), "C") && !(name in [:Y, :Test, :Id]) #if not categorical nor Test/Y
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
	for missingname in Symbol.(names(Z))
		for name in setdiff(Symbol.(names(newdf)), [:Test, :Y, :Id])
			result[!, Symbol("$(name)_$missingname")] = newdf[!, name] .* Z[!, missingname]
		end
	end
	if removezerocols
		result = removezerocolumns(result)
	end
	return result
end
