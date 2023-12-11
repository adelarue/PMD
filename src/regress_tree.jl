###################################
### regress_tree.jl
### Functions to perform regression with CART with MIA
### Authors: XXXX

###################################
# using DecisionTree

# """
# 	Fit a DT to the training data
# """
# function regress_tree(Y::Union{Vector{Float64},BitArray}, df::DataFrame; maxdepth::Int=5)
# 	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
# 	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
# 	X = Matrix{Float64}(df[trainingset, cols])
# 	y = convert(Array{Float64}, Y[trainingset])
# 	return DecisionTree.build_tree(y[:], X, 0, maxdepth, 1)
# end

# """
# 	Get Regression Tree predictions on dataset
# """
# function predict(df::DataFrame, model::DecisionTree.Root)
# 	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
# 	X = Matrix{Float64}(df[:, cols])
# 	return try DecisionTree.apply_tree_proba(model, X, ["1"])[:,1] catch ; DecisionTree.apply_tree(model, X) end
# 	# return try DecisionTree.apply_tree(model, X) catch ; DecisionTree.apply_tree_proba(model, X, ["1"])[:,1] end
# end


using ScikitLearn, PyCall
@sk_import tree: DecisionTreeRegressor
@sk_import tree: DecisionTreeClassifier

"""
	Fit a DTree to the training data
"""
function regress_tree(Y::Vector{Float64}, df::DataFrame; maxdepth::Int=5)

	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	X = Matrix{Float64}(df[trainingset, cols])
	y = convert(Array{Float64}, Y[trainingset])

	model = DecisionTreeRegressor(max_depth=maxdepth)

	ScikitLearn.fit!(model, X, y)

	return model
end
function regress_tree(Y::BitArray, df::DataFrame; maxdepth::Int=5)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	X = Matrix{Float64}(df[trainingset, cols])
	y = convert(Array{Float64}, Y[trainingset])

	model = DecisionTreeClassifier(max_depth=maxdepth)

	ScikitLearn.fit!(model, X, y)

	return model
end

"""
	Get DTree predictions on dataset
"""
function predict(df::DataFrame, model::PyCall.PyObject)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	X = Matrix{Float64}(df[:, cols])
	return (try ScikitLearn.predict_proba(model, X)[:,1] catch ; ScikitLearn.predict(model, X) end)
end
