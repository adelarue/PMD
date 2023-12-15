###################################
### regress_rf.jl
### Functions to perform regression with RF
### Authors: XXXX

###################################
# using DecisionTree
# using Random 

# threadsafe_rng = Random.TaskLocalRNG()

# """
# 	Fit a RF to the training data
# """
# function regress_rf(Y::Union{Vector{Float64},BitArray}, df::DataFrame; 
# 	maxdepth::Int=10, ntrees=10, nfeat::Int=round(Int, sqrt(Base.size(df,2))), psamples=1.)

# 	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
# 	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
# 	X = Matrix{Float64}(df[trainingset, cols])
# 	y = convert(Array{Float64}, Y[trainingset])
# 	return DecisionTree.build_forest(y[:], X, min(max(nfeat,1), length(cols)), ntrees, psamples, maxdepth, rng=threadsafe_rng)
# end

# """
# 	Get Random Forest predictions on dataset
# """
# function predict(df::DataFrame, model::DecisionTree.Ensemble)
# 	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
# 	X = Matrix{Float64}(df[:, cols])
# 	return try DecisionTree.apply_forest_proba(model, X, ["1"])[:,1] catch ; DecisionTree.apply_forest(model, X) end
# 	# return try DecisionTree.apply_tree(model, X) catch ; DecisionTree.apply_tree_proba(model, X, ["1"])[:,1] end
# end

# using PyCall
# using ScikitLearn: @sk_import, fit!, predict, predict_proba
# # @sk_import ensemble: RandomForestRegressor
# # @sk_import ensemble: RandomForestClassifier

# const RandomForestRegressor = PyNULL()
# const RandomForestClassifier = PyNULL()

# function __init__()
#     @eval @sk_import ensemble: RandomForestRegressor
# 	@eval @sk_import ensemble: RandomForestClassifier
# end


"""
	Fit a RF to the training data
"""
function regress_rf(Y::Vector{Float64}, df::DataFrame; 
	maxdepth::Int=10, ntrees=10)

	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	X = Matrix{Float64}(df[trainingset, cols])
	y = convert(Array{Float64}, Y[trainingset])

	model = RandomForestRegressor(n_estimators=ntrees, max_depth=maxdepth)

	ScikitLearn.fit!(model, X, y)

	return model
end
function regress_rf(Y::BitArray, df::DataFrame; 
	maxdepth::Int=10, ntrees=10)

	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	X = Matrix{Float64}(df[trainingset, cols])
	y = convert(Array{Float64}, Y[trainingset])

	model = RandomForestClassifier(n_estimators=ntrees, max_depth=maxdepth)

	ScikitLearn.fit!(model, X, y)

	return model
end

# """
# 	Get Random Forest predictions on dataset
# """
# function predict(df::DataFrame, model::PyCall.PyObject)
# 	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
# 	X = Matrix{Float64}(df[:, cols])
# 	return (try ScikitLearn.predict_proba(model, X)[:,1] catch ; ScikitLearn.predict(model, X) end)
# end