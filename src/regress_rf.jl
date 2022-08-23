###################################
### regress_rf.jl
### Functions to perform regression with RF
### Authors: Arthur Delarue, Jean Pauphilet, 2022
###################################
using DecisionTree

"""
	Fit a RF to the training data
"""
function regress_rf(Y::Union{Vector{Float64},BitArray}, df::DataFrame; 
	maxdepth::Int=10, ntrees=10, nfeat::Int=round(Int, sqrt(Base.size(df,2))), psamples=0.5)

	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	X = Matrix{Float64}(df[trainingset, cols])
	y = convert(Array{Float64}, Y[trainingset])
	return DecisionTree.build_forest(y[:], X, min(max(nfeat,1), length(cols)), ntrees, psamples, maxdepth)
end

"""
	Get Random Forest predictions on dataset
"""
function predict(df::DataFrame, model::DecisionTree.Ensemble)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	X = Matrix{Float64}(df[:, cols])
	return try DecisionTree.apply_forest_proba(model, X, ["1"])[:,1] catch ; DecisionTree.apply_forest(model, X) end
	# return try DecisionTree.apply_tree(model, X) catch ; DecisionTree.apply_tree_proba(model, X, ["1"])[:,1] end
end
