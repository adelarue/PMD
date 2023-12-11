################################################################################
# PHD module
################################################################################

module PHD

using SparseArrays, DataFrames, CSV
using RDatasets
using RCall
using GLMNet
using Statistics, LinearAlgebra, Printf, Random
using DataStructures, StatsBase, MLUtils
using JuMP, Gurobi
using Flux
using XGBoost
using PyCall, ScikitLearn
# using DecisionTree

"Helper function to load R packages and install them if necessary"
function load_R_library(name::AbstractString)
	try
		reval("library($name)")
	catch
		reval("install.packages('$name')")
		reval("library($name)")
	end
end

include("creation/tools.jl")
include("creation/uci.jl")
include("creation/rdatasets.jl")

include("count.jl")
include("impute.jl")
include("augment.jl")
include("generate_x.jl")
include("generate_y.jl")
include("regress_linear.jl")
include("regress_nn.jl")

using PyCall
using ScikitLearn: @sk_import, fit!, predict, predict_proba

const RandomForestRegressor = PyNULL()
const RandomForestClassifier = PyNULL()
const DecisionTreeRegressor = PyNULL()
const DecisionTreeClassifier = PyNULL()

function __init__()
	@eval @sk_import tree: DecisionTreeRegressor
	@eval @sk_import tree: DecisionTreeClassifier
    @eval @sk_import ensemble: RandomForestRegressor
	@eval @sk_import ensemble: RandomForestClassifier
end
include("regress_tree.jl")
include("regress_rf.jl")

include("regress_xgb.jl")
include("greedy.jl")
include("evaluate.jl")
include("validate.jl")
include("imp_then_reg.jl")

end # module