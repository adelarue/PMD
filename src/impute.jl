###################################
### impute.jl
### Functions to impute missing values before performing regression
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Standardize column names R-style
"""
function standardize_colnames(df::DataFrame)
	aux = select(df, Not(:Id))
	load_R_library("dplyr")
	R"train = $aux"
	R"colnames <- names(train)"
	R"names(train) <- make.names(colnames, unique=TRUE)"
	@rget train
	train[!,:Id] = df[:,:Id]
	return train
end

"""
	Impute missing values in a dataset using mice
"""
function mice(df::DataFrame; m_imputation=2, max_epoch=5)
	aux = select(df, Not(:Id))
	load_R_library("mice")
	load_R_library("dplyr")
	R"train = $aux"

	R"colnames <- names(train)"
	R"names(train) <- make.names(colnames, unique=TRUE)"
	try
		R"imputed = mice(as.data.frame(train), m=$m_imputation, maxit=$max_epoch, printFlag=F, seed=4326)"
	catch
		R"imputed = mice(as.data.frame(train), m=$m_imputation, maxit=$max_epoch, printFlag=F, seed=4326, method='cart')"
	end
	R"imputedtrain = complete(imputed, action=1)"
	R"names(imputedtrain) <- colnames"
	@rget imputedtrain
	imputedtrain[!,:Id] = df[:,:Id]
	# R"trainplus = select(as.data.frame(df), -c(Test))"
	# # for some reason, sometimes columns are lists
	# R"trainplus = t(apply(trainplus, 1, unlist))"
	# R"imputedplus = mice(as.data.frame(trainplus), m=1, printFlag=F)"
	# R"imputedtest = select(subset(mutate(complete(imputedplus, action=1), Test=df$Test), Test==1), -Test)"
	# @rget imputedtest
	# result[result.Test .== 0, setdiff(names(result), [:Test])] .= imputedtrain
	# result[result.Test .== 1, setdiff(names(result), [:Test])] .= imputedtest
	return imputedtrain
end

function mice_bruteforce(df::DataFrame; m_imputation=2, max_epoch=5)
	df_imputed = PHD.mice(df, m_imputation=m_imputation, max_epoch=max_epoch)
	if any([mean(ismissing.(df_imputed[:,k])) > 0 for k in names(df)]) #If some columns are still missing
        df_imputed = PHD.mice(df_imputed, m_imputation=5, max_epoch=30) #Reimpute with more effort
    end
    if any([mean(ismissing.(df_imputed[:,k])) > 0 for k in names(df)]) #If still some columns are still missing
        select!(df_imputed, Not([k for k in names(df) if mean(ismissing.(df_imputed[:,k])) > 0])) #Drop
    end
	return df_imputed
end

"""
	Impute all missing values as mean -- Utilities
"""
function compute_mean(df::DataFrame)
	numcols = filter(t-> string(t) ∉ ["Id", "Test"], names(df))
	nummeans = []
	for c in numcols
	    push!(nummeans, mean(skipmissing(df[:,c])))
	end
	return DataFrame(nummeans', numcols)
end
function meanvector_to_df(μ, numcols)
	r = DataFrame(μ', numcols)
	if :Test ∈ numcols || "Test" ∈ numcols
		select(r, Not(:Test))
	else
		r
	end
end

"""
	Impute all missing values by `means` (DataFrame)
"""
function mean_impute(df::DataFrame, means::DataFrame)
	result = deepcopy(df)
	for n in names(means)
		result[!,n] = convert(Array{Union{Missing,Float64},1},result[:,n])
		result[ismissing.(result[:,n]),n] .= means[1,n]
	end
	return result
end

"""
	Impute all missing values by `μ` (Vector) (Optimized)
"""
function mu_impute(df::DataFrame, μ; missing_columns=collect(1:Base.size(df,1)))
	result = deepcopy(df)
	for i in missing_columns
		result[!,i] = convert(Array{Union{Missing,Float64},1},result[:,i])
		result[ismissing.(df[:,i]),i] .= μ[i]
	end
	return result
end
function update_mu_impute!(df::DataFrame, j::Union{Int,String}, μj, M)
	df[M[:,j] .== 1,j] .= μj
end

"""
	Impute all missing values as zeros
"""
function zeroimpute!(df::DataFrame)
	for n in names(df)
		if sum(ismissing.(df[:,n])) > 0
			df[!,n] = convert(Array{Union{Missing,Float64},1},df[:,n])
			df[ismissing.(df[:,n]),n] .= 0
		end
	end
end
function zeroimpute(df::DataFrame)
	result = deepcopy(df)
	zeroimpute!(result)
	return result
end

"""
	Remove indicator of missingness for categorical - Mode impute instead
"""
function mode_impute!(df::DataFrame; train=trues(Base.size(df,1)), deencode_only=true)
	onehotencoded_missing = [n for n in names(df) if endswith(String(n),"_Missing")]
	for f in onehotencoded_missing
	    feat = split(String(f), "_Missing", keepempty=false)[1]
	    onehot_feat = [n for n in names(df) if startswith(String(n), feat) && !endswith(String(n),"_Missing")]
		if length(onehot_feat) > 0
			if deencode_only
				freq = vec(mean(Matrix(df[train,onehot_feat]), dims=1))
		    	imax = argmax(freq)
		    	df[df[:,f] .== 1, onehot_feat[imax]] .= 1
			else
				for n in onehot_feat
					df[!,n] = convert(Array{Union{Missing,Float64},1},[df[i,n] for i in 1:Base.size(df,1)])
					df[df[:,f] .== 1, n] .= missing
				end
			end
		end
	end
	select!(df, Not(onehotencoded_missing))
end
