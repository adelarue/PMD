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
using DecisionTree

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
include("regress_tree.jl")
include("regress_rf.jl")
include("greedy.jl")
include("evaluate.jl")
include("validate.jl")
include("imp_then_reg.jl")

end # module