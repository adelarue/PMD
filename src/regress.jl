###################################
### regress.jl
### Functions to perform linear regression
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Perform linear regression on a dataframe
	Args:
		- a dataframe with no missing values
	Parameters:
		- lasso: true if we use lasso (penalty chosen using cross-validation),
				 false if we do not regularize
		- alpha: elastic net parameter (1.0 for lasso, 0.0 for ridge)
		- missing_penalty: how much to penalize augmented columns
				 (columns that contain missing in their name). Regular columns have penalty 1.0
	Returns:
		- a dataframe with a single row, with the same column names as the data
		except Test, and an additional Offset containing the constant term
"""

using MLDataPattern
R"""library(genlasso)"""
R"""options(warn = - 1) """
function genlassocv(X, y, D; kfolds =10,method=:kcv)
	R"""library(genlasso)"""
	R"""options(warn = - 1) """

    R"""genlassopath <- genlasso($y, as.matrix($X), ($D), verbose=F, eps=0.1)"""
    @rget genlassopath
    
    λ = genlassopath[:lambda]
    oo_performance = zeros(length(λ),kfolds)
    
    i_fold = 0
    for ((Xtrain, ytrain), (Xval, yval)) in MLDataPattern.kfolds((X',y), k=kfolds)
        i_fold += 1
        Xtrain = Xtrain'; Xval = Xval'
        R"""modeltrain <- genlasso($ytrain, as.matrix($Xtrain), ($D), verbose=F, eps=0.1)
            betatrain = coef(modeltrain, $λ)
        """
        @rget betatrain
        pred_val = Xval*betatrain[:beta] 
        oo_performance[:,i_fold] .= sum((pred_val .- yval*ones(length(λ))').^2, dims=1)[:]
		# if length(unique(yval)) == 2
		# 	for lval = 1:length(λ)
		# 		oo_performance[lval,i_fold] = 1-auc(convert(BitArray, yval), pred_val[:,lval])
		# 	end
		# end
    end

    genlassopath[:oo_mse] = mean(oo_performance,dims=2)[:]
    
    return genlassopath
end


# #KEPT FOR COMPATIBILITY: CHOOSE REG WITH BOOL ARG "lasso", NOW REPLACED BY SYMBOL ARG "regtype"
# function regress(Y::Array{Float64}, df::DataFrame;
# 	lasso::Bool=false, alpha::Real=0.8, missing_penalty::Real=1.0)
# 	return regress(Y, df; regtype= lasso ? :lasso : :none, alpha=alpha, missing_penalty=missing_penalty)
# end

using RCall, SparseArrays
# R"""install.packages("genlasso")"""
R"""library(genlasso)"""

function regress(Y::Array{Float64}, df::DataFrame;
				regtype::Symbol=:none, alpha::Real=0.8, missing_penalty::Real=1.0)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	X = Matrix(df[df[:, :Test] .== 0, cols])
	y = convert(Array, Y[df[:,:Test] .== 0])
	coefficients = DataFrame()
	if regtype == :lasso
		try
			penalty_factor = ones(length(cols))
			for (i, col) in enumerate(cols)
				if occursin("_missing", string(col))
					penalty_factor[i] = missing_penalty
				end
			end
			cv = glmnetcv(X, y, alpha=alpha, penalty_factor=penalty_factor)
			for (i, col) in enumerate(cols)
				coefficients[!,col] = ([cv.path.betas[i, argmin(cv.meanloss)]])
			end
			coefficients[!,:Offset] = [cv.path.a0[argmin(cv.meanloss)]]
		catch
			println("Error in regression")
			for col in cols
				coefficients[!,col] .= 0.
			end
			coefficients[!,:Offset] .= mean(y)
		end
	elseif regtype == :missing_weight
		penalty_factor = ones(length(cols))
		for (i, col) in enumerate(String.(cols))
			if occursin("_missing", string(col))
				penalty_factor[i] = missing_penalty
			end
			p = mean(X[:,i] .== 0)
			penalty_factor[i] *= p
		end
		cv = glmnetcv(X, y, alpha=alpha, penalty_factor=penalty_factor)
		for (i, col) in enumerate(cols)
			coefficients[!,col] = ([cv.path.betas[i, argmin(cv.meanloss)]])
		end
		coefficients[!,:Offset] = [cv.path.a0[argmin(cv.meanloss)]]
	elseif regtype == :genlasso
		cols = union(["Offset"], String.(cols))
		X = [ones(Base.size(X,1)) X] #Adding intercept because genlasso does not support
		unique_cols = intersect(unique(map(t -> replace(t, "_missing" => ""), cols)),cols)
		counter = 0 
		D_list = []
		for j in setdiff(unique_cols, ["Offset"])
			kj = findfirst(cols .== j)    
			#Lasso penalty on feature j (x_j)
			counter += 1
			kj_influenced = findall( map(t -> startswith(t, j), cols) )
			if length(kj_influenced) == 0
				@show j
			end 
			for i in kj_influenced
				push!(D_list, (counter, i, 1))
			end

			#Lasso penalty on feature j missing (m_j)
			kj_missing = findall( map(t -> endswith(t, j*"_missing"), cols) )
			if length(kj_missing) > 0 && missing_penalty > 0
				counter += 1
				for i in kj_missing
					push!(D_list, (counter, i, -alpha*missing_penalty))
				end
				for i in setdiff(kj_influenced,kj_missing)
					push!(D_list, (counter, i, missing_penalty))
				end
			end
		end
		D  = sparse([d[1] for d in D_list],[d[2] for d in D_list],[Float64(d[3]) for d in D_list], counter, Base.size(X,2))

		# # @rput y X D
		# R"""library(genlasso)
		# 	genlassopath <- genlasso($y, as.matrix($X), ($D), verbose=F, eps=0.1, svd=T)"""
		# @rget genlassopath
		
		genlassopath = genlassocv(X, y, D)
		best_lambda = argmin(genlassopath[:oo_mse])

		for (i, col) in enumerate(cols)
			coefficients[!,col] = [genlassopath[:beta][i,best_lambda]]
		end
	else
		path = glmnet(X, y)
		for (i, col) in enumerate(cols)
			coefficients[!,col] = [path.betas[i, end]]
		end
		coefficients[!,:Offset] = [path.a0[end]]
	end
	coefficients[!,:Logistic] = [false]
	return coefficients
end

# function regress(Y::BitArray{1}, df::DataFrame;
# 	lasso::Bool=false, alpha::Real=0.8, missing_penalty::Real=1.0)
# 	return regress(Y, df; regtype= lasso ? :lasso : :none, alpha=alpha, missing_penalty=missing_penalty)
# end
function regress(Y::BitArray{1}, df::DataFrame;
				 regtype::Symbol=:lasso, alpha::Real=0.8, missing_penalty::Real=1.0)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	X = Matrix(df[df[:, :Test] .== 0, cols])
	y = convert(Array, Y[df[!, :Test] .== 0])

	freq = mean(1.0.*y)
	w = (1/freq).*y .+ (1/(1 - freq)).*(1. .- y); #class weights
	coefficients = DataFrame()
	if regtype == :lasso
		# try
			penalty_factor = ones(length(cols))
			for (i, col) in enumerate(cols)
				if occursin("_missing", string(col))
					penalty_factor[i] = missing_penalty
				end
			end
			cv = glmnetcv(X, hcat(Float64.(.!y), Float64.(y)), GLMNet.Binomial(),
			              alpha=alpha, penalty_factor=penalty_factor, weights=w)
			for (i, col) in enumerate(cols)
				coefficients[!,col] = [cv.path.betas[i, argmin(cv.meanloss)]]
			end
			coefficients[!,:Offset] = [cv.path.a0[argmin(cv.meanloss)]]
		# catch
		# 	for col in cols
		# 		coefficients[!,col] = [0.]
		# 	end
		# 	coefficients[!,:Offset] = [log(mean(y) / (1-mean(y))]
		# end
	elseif regtype == :missing_weight
		penalty_factor = ones(length(cols))
		for (i, col) in enumerate(String.(cols))
			if occursin("_missing", string(col))
				penalty_factor[i] = missing_penalty
			end
			p = mean(X[:,i] .== 0)
			penalty_factor[i] *= p
		end
		cv = glmnetcv(X, hcat(Float64.(.!y), Float64.(y)), GLMNet.Binomial(),
						alpha=alpha, penalty_factor=penalty_factor, weights=w)
		for (i, col) in enumerate(cols)
			coefficients[!,col] = [cv.path.betas[i, argmin(cv.meanloss)]]
		end
		coefficients[!,:Offset] = [cv.path.a0[argmin(cv.meanloss)]]
	elseif regtype == :genlasso
		coefficients = regress(1.0*Y, df; regtype=:genlasso, alpha=alpha, missing_penalty=missing_penalty)
		select!(coefficients, Not(:Logistic))
	else
		path = glmnet(X, hcat(Float64.(.!y), Float64.(y)), GLMNet.Binomial(), weights=w)
		for (i, col) in enumerate(cols)
			coefficients[!,col] = [path.betas[i, end]]
		end
		coefficients[!,:Offset] = [path.a0[end]]
	end
	coefficients[!, :Logistic] = [true]
	return coefficients
end

"""
	Predict the dependent variable given a regression model
"""
function predict(df::DataFrame, model::DataFrame)
	prediction = model[1, :Offset] .* ones(nrow(df))
	for name in setdiff(Symbol.(names(model)), [:Offset, :Logistic])
		prediction .+= (model[1, name]*df[:,name])
	end
	if model[1, :Logistic]
		return sigmoid.(prediction)
	else
		return prediction
	end
end

"""
	Evaluate the fit quality of a linear model on a dataset
"""
function evaluate(Y::Vector, df::DataFrame, model::DataFrame)
	prediction = predict(df, model)
	if model[1, :Logistic]
		error("Logistic model evaluated on continuous vector")
	else
		trainmean = Statistics.mean(Y[df[:,:Test] .== 0])
		SST = sum((Y[df[:,:Test] .== 0] .- trainmean) .^ 2)
		OSSST = sum((Y[df[:,:Test] .== 1] .- trainmean) .^ 2)

		R2 = 1 - sum((Y[df[:,:Test] .== 0] .- prediction[df[:,:Test] .== 0]) .^ 2)/SST
		OSR2 = 1 - sum((Y[df[:,:Test] .== 1] .- prediction[df[:,:Test] .== 1]) .^ 2)/OSSST
		return R2, OSR2
	end
end
function evaluate(Y::BitArray{1}, df::DataFrame, model::DataFrame;
				  metric::AbstractString="auc")
	prediction = predict(df, model)
	if model[1, :Logistic]
		if metric == "logloss"
			ll = logloss(Y[df[:,:Test] .== 0], prediction[df[:,:Test] .== 0])
			osll = logloss(Y[df[:,:Test] .== 1], prediction[df[:,:Test] .== 1])
			return ll, osll
		elseif metric == "auc"
			return auc(Y[df[:,:Test] .== 0], prediction[df[:,:Test] .== 0]),
				   auc(Y[df[:,:Test] .== 1], prediction[df[:,:Test] .== 1])
		else
			error("Unknown metric: $metric (only supports 'logloss', 'auc')")
		end
	else
		error("Continuous model evaluated on binary vector")
	end
end

"Compute logistic loss of a particular prediction"
function safelog(x)
	if x < 1e-10
		return log(1e-10)
	elseif x > 1e10 
		return log(1e10)
	else 
		return log(x)
	end
end
function logloss(actual::BitArray{1}, predicted::Vector)
	return sum(actual .* safelog.(predicted) .+ (1 .- actual) .* safelog.(1 .- predicted))
end
logloss(actual::BitArray{1}, constant::Real) = logloss(actual, constant .* ones(length(actual)))

"Compute AUC of predictions"
function auc(actual::BitArray{1}, predicted::Vector)
	@assert length(actual) == length(predicted)
	r = StatsBase.tiedrank(predicted)
	n_pos = sum(actual)
	n_neg = length(actual) - n_pos
	AUC = (sum(r[actual]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
	return AUC
end

"Transpose a linear model dataframe"
function transpose(linear)
	@assert nrow(linear) == 1
	return DataFrame(Feature = names(linear), Coeff = vec(linear |> Matrix))
end
