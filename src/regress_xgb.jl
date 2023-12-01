###################################
### regress_xgb.jl
### Functions to perform regression with RF
### Authors: XXXX

###################################
using XGBoost
using Suppressor 

"""
	Fit a XGB to the training data
"""
function regress_xgboost(Y::Union{Vector{Float64},BitArray}, df::DataFrame; 
    max_depth=6, min_child_weight=1.0,
    gamma=0., subsample=1.0, 
    colsample_bytree=1.0,
    alpha=0.0, lambda=1.0,
    n_estimators=100)

    loss = "reg:squarederror"
    if typeof(Y) == BitArray
        loss = "binary:logistic"
    end
    cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	X = Matrix(df[trainingset, cols])
	y = convert(Array, Y[trainingset])

    bst = []
    @suppress bst = XGBoost.xgboost(XGBoost.DMatrix(X, label=y), 
                                max_depth=max_depth, min_child_weight=min_child_weight,
                                gamma=gamma, subsample=subsample, 
                                colsample_bytree=colsample_bytree,
                                alpha=alpha, lambda=lambda,
                                n_estimators=n_estimators, 
                                verbosity = 0,
                                objective=loss)

	return bst 
end
# function regress_xgboost(Y::BitArray, df::DataFrame; 
#     max_depth=6, n_estimators=100, gamma=0.2)

#     cols = setdiff(Symbol.(names(df)), [:Id, :Test])
# 	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
# 	X = Matrix(df[trainingset, cols])
# 	y = convert(Array, Y[trainingset])

#     # bst = XGBoost.Booster(XGBoost.DMatrix(X, label=y), max_depth=max_depth, η=0.5, objective="binary:logistic")

#     # @suppress XGBoost.update!(bst, XGBoost.DMatrix(X, label=y), num_round=n_estimators)
#     bst = []
#     @suppress bst = XGBoost.xgboost(XGBoost.DMatrix(X, label=y), gamma=gamma, max_depth=max_depth, subsample=0.8, num_round=n_estimators, objective="binary:logistic")

# 	return bst 
# end
"""
	Get XGB predictions on dataset
"""
function predict(df::DataFrame, model::XGBoost.Booster)
    cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	X = (df[:, cols])

    return XGBoost.predict(model, XGBoost.DMatrix(X))
end
