###################################
### regress.jl
### Functions to perform linear regression
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################


using RCall, SparseArrays
using MLUtils

# R"""install.packages("genlasso")"""
R"""library(genlasso)"""
R"""options(warn = - 1) """
"""
	Construct the equivalent of glmnetcv but for Generalized Lasso penalties
"""
function genlassocv(X, y, D; kfolds =10,method=:kcv)
	R"""library(genlasso)"""
	R"""options(warn = - 1) """

    R"""genlassopath <- genlasso($y, as.matrix($X), ($D), verbose=F, eps=0.1)"""
    @rget genlassopath
    
    λ = genlassopath[:lambda]
    oo_performance = zeros(length(λ),kfolds)
    
    i_fold = 0
    for ((Xtrain, ytrain), (Xval, yval)) in MLUtils.kfolds((X',y), k=kfolds)
        i_fold += 1
        Xtrain = Xtrain'; Xval = Xval'
        R"""modeltrain <- genlasso($ytrain, as.matrix($Xtrain), ($D), verbose=F, eps=0.1)
            betatrain = coef(modeltrain, $λ)
        """
        @rget betatrain
        pred_val = Xval*betatrain[:beta] 
        oo_performance[:,i_fold] .= sum((pred_val .- yval*ones(length(λ))').^2, dims=1)[:]
    end

    genlassopath[:oo_mse] = mean(oo_performance,dims=2)[:]
    
    return genlassopath
end



"""
	Perform Lasso-type linear regression on a dataframe. 
	λ hyper-parameter is CVed by default
	α (1-ridge term) and missing_penalty (λ scaling factor for the m_j features) are not
	
	Args:
		- a target variable Y
		- a dataframe with no missing values
	Parameters:
		- regtype: type of penalty. One of: 
			:lasso (regular Lasso), 
			:missing_weight (Lasso where λ is scaled by proportion of missingness for each feature)
			:genlasso (generalized Lasso penalty accounting for multi-colinearity) 
		- alpha: elastic net parameter (1.0 for lasso, 0.0 for ridge)
		- missing_penalty: how much to penalize augmented columns
				 (columns that contain missing in their name). Regular columns have penalty 1.0
	Returns:
		- a linear regression model, encoded as a single-row dataframe, with 
			the same column names as the data - :Test + :Offset
"""
function regress_linear(Y::Array{Float64}, df::DataFrame;
				regtype::Symbol=:missing_weight, alpha::Real=0.8, missing_penalty::Real=1.0)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	X = Matrix(df[trainingset, cols])
	y = convert(Array, Y[trainingset])
	coefficients = DataFrame()
	
	#PENALTY: Lasso 
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
	
	#PENALTY: Lasso scaled by fraction of missing
	elseif regtype == :missing_weight
		penalty_factor = ones(length(cols))
		for (i, col) in enumerate(String.(cols))
			p = mean(X[:,i] .== 0)
			penalty_factor[i] += ( occursin("_missing", string(col)) ? missing_penalty : 1)*p
		end
		cv = glmnetcv(X, y, alpha=alpha, penalty_factor=penalty_factor)
		for (i, col) in enumerate(cols)
			coefficients[!,col] = ([cv.path.betas[i, argmin(cv.meanloss)]])
		end
		coefficients[!,:Offset] = [cv.path.a0[argmin(cv.meanloss)]]
	
	#PENALTY: Generalized Lasso--Not useful
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
		
		genlassopath = genlassocv(X, y, D)
		best_lambda = argmin(genlassopath[:oo_mse])

		for (i, col) in enumerate(cols)
			coefficients[!,col] = [genlassopath[:beta][i,best_lambda]]
		end
	#Default option: Lasso with λ not CVed	
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

#Binary classification version
function regress_linear(Y::BitArray{1}, df::DataFrame;
				 regtype::Symbol=:lasso, alpha::Real=0.8, missing_penalty::Real=1.0)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	X = Matrix(df[trainingset, cols])
	y = convert(Array, Y[trainingset])

	freq = mean(1.0.*y)
		
	w = (1/freq).*y .+ (1/(1 - freq)).*(1. .- y); #class weights
	coefficients = DataFrame()
	
	#PENALTY: Lasso 
	if freq > 0.999 || freq < 0.111
		coefficients[!,:Offset] = [safelog(mean(y) / (1-mean(y)))]
	elseif regtype == :lasso
		penalty_factor = ones(length(cols))
		for (i, col) in enumerate(cols)
			if occursin("_missing", string(col))
				penalty_factor[i] = missing_penalty
			end
		end
		cv = glmnetcv(X, hcat(Float64.(.!y), Float64.(y)), GLMNet.Binomial(),
						alpha=alpha, penalty_factor=penalty_factor, weights=w)
		if length(cv.meanloss) > 0
			for (i, col) in enumerate(cols)
				coefficients[!,col] = [cv.path.betas[i, argmin(cv.meanloss)]]
			end
			coefficients[!,:Offset] = [cv.path.a0[argmin(cv.meanloss)]]
		else
			for col in cols
				coefficients[!,col] = [0.]
			end
			coefficients[!,:Offset] = [ log( mean(y) / (1-mean(y))) ]
		end
	
	#PENALTY: Lasso scaled by fraction of missing
	elseif regtype == :missing_weight
		penalty_factor = ones(length(cols))
		for (i, col) in enumerate(String.(cols))
			p = mean(X[:,i] .== 0)
			penalty_factor[i] += ( occursin("_missing", string(col)) ? missing_penalty : 1)*p
		end
		cv = glmnetcv(X, hcat(Float64.(.!y), Float64.(y)), GLMNet.Binomial(),
						alpha=alpha, penalty_factor=penalty_factor, weights=w)
		if length(cv.meanloss) > 0
			for (i, col) in enumerate(cols)
				coefficients[!,col] = [cv.path.betas[i, argmin(cv.meanloss)]]
			end
			coefficients[!,:Offset] = [cv.path.a0[argmin(cv.meanloss)]]
		else
			for col in cols
				coefficients[!,col] = [0.]
			end
			coefficients[!,:Offset] = [log(mean(y) / (1-mean(y)))]
		end

	#PENALTY: Generalized Lasso--Not useful
	elseif regtype == :genlasso
		coefficients = regress(1.0*Y, df; regtype=:genlasso, alpha=alpha, missing_penalty=missing_penalty)
		select!(coefficients, Not(:Logistic))
	
	#Default option: Lasso with λ not CVed	
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


"Transpose a linear model dataframe"
function transpose(linear)
	@assert nrow(linear) == 1
	return DataFrame(Feature = names(linear), Coeff = vec(linear |> Matrix))
end
