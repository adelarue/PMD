################################################################################
# PHD module
################################################################################

module PHD

using IrrationalConstants,SparseArrays, DataFrames, CSV
using RDatasets
using RCall
using GLMNet
using Statistics, LinearAlgebra, Printf, Random
using DataStructures, StatsBase, MLDataPattern
using JuMP, Gurobi
using Flux

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
include("generate_y.jl")
include("regress.jl")
include("regress_nonlinear.jl")
include("regress_tree.jl")
include("greedy.jl")
include("validate.jl")

end # module