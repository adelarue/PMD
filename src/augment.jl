###################################
### augment.jl
### Functions to add PHD features
### Authors: XXXX
###################################

"""
	Recover the mu-vector from a static with affine intercept model
"""
function recover_mu(linear::DataFrame, canbemissing::Vector{String})
	μ = []
	for j in canbemissing
		push!(μ, linear[1,j*"_missing"] / linear[1,j])
	end
	μ
end

"""
	Remove columns that are uniformly zero
"""
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
			result[!,Symbol("$(name)_missing")] = (Int.(ismissing.(df[:,name])))
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
			if name != missingname #No self-correction - should be captured bu removecols step as well
				result[!, Symbol("$(name)_$missingname")] = newdf[:, name] .* Z[:, missingname]
			end
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
	Do ±Inf imputation for missingness (MIA-trick)
"""
function augment_MIA(df; bigM=1e10)
    M = indicatemissing(df, removecols=:Constant)
    cols_with_missing = names(M); map!(t -> split(t, "_missing")[1], cols_with_missing, cols_with_missing)
    cols_never_missing = setdiff(names(df), cols_with_missing)
    #Replace missing by +Inf
	imp_plus = df[:,cols_with_missing]
    for name in cols_with_missing
        imp_plus[ismissing.(imp_plus[:,name]),name] .= convert(eltype(imp_plus[:,name]), bigM)
    end
    rename!(imp_plus, [(i => i*"_plus") for i in cols_with_missing])
	#Replace missing by -Inf
    imp_neg = df[:,cols_with_missing]
    for name in cols_with_missing
        imp_neg[ismissing.(imp_neg[:,name]),name] .= convert(eltype(imp_neg[:,name]), -bigM)
    end
    rename!(imp_neg, [(i => i*"_minus") for i in cols_with_missing])
    
    return hcat(imp_plus, imp_neg, df[:,cols_never_missing])
end