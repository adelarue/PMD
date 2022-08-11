###################################
### regress_tree.jl
### Functions to perform regression with CART with MIA
### Authors: Arthur Delarue, Jean Pauphilet, 2022
###################################
using DecisionTree

# """
# 	Fit a DT to the training data
# """
# function regress_tree_cv(Y::Vector{Float64}, data::DataFrame; 
# 	maxdepthlist::Array{Int}=collect(1:2:10), val_fraction::Real=0.2)

# 	newY = Y[data[:,:Test] .== 0]
# 	newdata = filter(row -> row[:Test] == 0, data)

# 	val_indices = findall(split_dataset(newdata; test_fraction = val_fraction, random=true))
# 	newdata[val_indices, :Test] .= 1

# 	bestmodel = regress_tree(newY, newdata, maxdepth=maxdepthlist[1])
# 	bestOSR2 = evaluate(newY, newdata, bestmodel)[2]
# 	bestparams = maxdepthlist[1]

# 	for d in maxdepthlist
# 		newmodel = regress_tree(newY, newdata, maxdepth = d)
# 		newOSR2 = evaluate(newY, newdata, newmodel)[2]
# 		if newOSR2 > bestOSR2
# 			bestOSR2 = newOSR2
# 			bestparams = d
# 		end
# 	end

# 	bestmodel = regress_tree(Y, data, maxdepth=bestparams)
# 	return bestmodel, bestparams
# end

"""
	Fit a DT to the training data
"""
function regress_tree(Y::Union{Vector{Float64},BitArray}, df::DataFrame; maxdepth::Int=5)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	X = Matrix{Float64}(df[trainingset, cols])
	y = convert(Array{Float64}, Y[trainingset])
	return DecisionTree.build_tree(y[:], X, 0, maxdepth, 1)
end

"""
	Get Regression Tree predictions on dataset
"""
# function predict(df::DataFrame, model::DecisionTree.Root)
# 	predict(df, model.node)
# end
function predict(df::DataFrame, model::DecisionTree.Root)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	X = Matrix{Float64}(df[:, cols])
	return try DecisionTree.apply_tree_proba(model, X, ["1"])[:,1] catch ; DecisionTree.apply_tree(model, X) end
	# return try DecisionTree.apply_tree(model, X) catch ; DecisionTree.apply_tree_proba(model, X, ["1"])[:,1] end
end
# function predict_proba(df::DataFrame, model::DecisionTree.Node)
# 	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
# 	X = Matrix(df[:, cols])
# 	return DecisionTree.apply_tree_proba(model, X, ["1"])[:,1]
# end

# """
# 	Evaluate fit quality on dataset
# """
# function evaluate(Y::Vector{Float64}, df::DataFrame, model::DecisionTree.Node)
# 	prediction = predict(df, model)
# 	trainmean = Statistics.mean(Y[df[:,:Test] .== 0])
# 	SST = sum((Y[df[:,:Test] .== 0] .- trainmean) .^ 2)
# 	OSSST = sum((Y[df[:,:Test] .== 1] .- trainmean) .^ 2)
# 	R2 = 1 - sum((Y[df[:,:Test] .== 0] .- prediction[df[:,:Test] .== 0]) .^ 2)/SST
# 	OSR2 = 1 - sum((Y[df[:,:Test] .== 1] .- prediction[df[:,:Test] .== 1]) .^ 2)/OSSST
# 	return R2, OSR2
# end
# function evaluate(Y::BitArray{1}, df::DataFrame, model::DecisionTree.Node; metric::AbstractString="auc")
# 	prediction = predict_proba(df, model)
# 	if metric == "logloss"
# 		ll = logloss(Y[df[:,:Test] .== 0], prediction[df[:,:Test] .== 0])
# 		osll = logloss(Y[df[:,:Test] .== 1], prediction[df[:,:Test] .== 1])
# 		return ll, osll
# 	elseif metric == "auc"
# 		return auc(Y[df[:,:Test] .== 0], prediction[df[:,:Test] .== 0]),
# 						auc(Y[df[:,:Test] .== 1], prediction[df[:,:Test] .== 1])
# 	else
# 		error("Unknown metric: $metric (only supports 'logloss', 'auc')")
# 	end
# end