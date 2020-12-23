###################################
### augment.jl
### Functions to add PHD features
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Remove columns that are uniformly zero
"""
# function removezerocolumns(df::DataFrame)
# 	tokeep = []
# 	for name in Symbol.(names(df))
# 		if name ∈ [:Id, :Test, :Y] || (!all(abs.(df[!,name]) .< 1e-9))
# 			push!(tokeep, name)
# 		end
# 	end
# 	return df[!, tokeep]
# end
function find_zerocolumns(df::DataFrame)
	return [n for n in Symbol.(names(df)) if n ∈ [:Id, :Test, :Y] || (!all(abs.(df[!,n]) .< 1e-9))]
end
function find_constantcolumns(df::DataFrame)
	return [n for n in Symbol.(names(df)) if n ∈ [:Id, :Test, :Y] || (var(df[:,n]) > 1e-9)]
end

"""
	Create the matrix M such that M_ij = 1 if X_ij is missing and 0 otherwise
"""
function indicatemissing(df::DataFrame; removecols::Symbol=:None)
	result = DataFrame()
	for name in Symbol.(names(df))
		if !startswith(String(name), "C") && !(name in [:Y, :Test, :Id]) #if not categorical nor Test/Y
			result[:,Symbol("$(name)_missing")] = (Int.(ismissing.(df[name])))
		end
	end

	if removecols == :None
		return result
	elseif removecols == :Zero
		return result[:,find_zerocolumns(result)]
	elseif removecols == :Constant
		return result[:,find_constantcolumns(result)]
	end
end

"""
	Create the outer product between the data matrix and the missingness indicators
"""
function augmentaffine(df::DataFrame; model::Array{String}=String.(names(df)), removecols::Symbol=:None)
	newdf = zeroimpute(df)
	Z = indicatemissing(df, removecols=:Constant)
	result = hcat(newdf, Z) #Start with zero imputation + offset adaptive rule
	for name in setdiff(intersect(String.(names(df)), model), ["Id", "Y", "Test"]) #W_{jk} where j is the feature from the model to correct
		for missingname in String.(names(Z)) 		#and k is the potentially missing feature triggering correction
			result[!, Symbol("$(name)_$missingname")] = newdf[:, name] .* Z[:, missingname]
		end
	end
	if removecols == :None
		return result
	elseif removecols == :Zero
		return result[:,find_zerocolumns(result)]
	elseif removecols == :Constant
		return result[:,find_constantcolumns(result)]
	end
end
