###################################################
## greedy.jl
##      Greedy tree-like method for regression
## Author: Arthur Delarue, Jean Pauphilet, 2019
###################################################

# import Base.<

abstract type Node end

"""
	A split node has two children, one where a particular feature is present (left),
		and the other where it is absent (right)
"""
struct SplitNode <: Node
	"Node id"
	id::Int
	"Node depth"
	depth::Int
	"Which feature we are splitting on"
	feature::String
	"Left child (feature present)"
	left::Int
	"Right child (feature absent)"
	right::Int
	# "Constant term in regression model"
	# intercept::Float64
	# "Regression coefficients"
	# coeffs::Vector{Float64}
	"Regression model"
	coeffs::Any
	"Current out-of-sample error"
	current_error::Float64
	"Parent node"
	parent::Int
end

"""
	A leaf node has no children, and it has a regression model attached to it
"""
struct LeafNode <: Node
	"Node id"
	id::Int
	"Node depth"
	depth::Int
	# "Constant term in regression model"
	# intercept::Float64
	# "Regression coefficients"
	# coeffs::Vector{Float64}
	"Regression model"
	model::Any
	"Features always present at the leaf"
	featuresIn::Vector{String}
	"Features always missing at the leaf"
	featuresOut::Vector{String}
	"Current out-of-sample error"
	currentError::Float64
	"Parent node"
	parent::Int
end

"""
	Regression model
"""
struct GreedyModel
	"Nodes of the tree"
	nodes::Vector{Node}
	"Leaf nodes"
	leaves::Set{Int}
	"Linear or logistic"
	logistic::Bool
end

"""
	Data structure for heap
"""
struct SplitCandidate
	"Leaf index"
	leaf::Int
	"Improvement to SSE from this split"
	improvement::Float64
	"New feature to split on"
	feature::String
	"Set of new left points"
	leftPoints::Vector{Int}
	"Set of new right points"
	rightPoints::Vector{Int}
	"Set of new left features"
	leftFeatures::Vector{String}
	"Set of new right features"
	rightFeatures::Vector{String}
	# "New left intercept"
	# leftIntercept::Float64
	# "New left coefficients"
	# leftCoeffs::Vector{Float64}
	# "New right intercept"
	# rightIntercept::Float64
	# "New right coeffs"
	# rightCoeffs::Vector{Float64}
	"New left model"
	leftModel::Any
	"New right model"
	rightModel::Any
	"New right out-of-sample error"
	rightError::Float64
	"New left out-of-sample error"
	leftError::Float64
end
Base.:<(e1::SplitCandidate, e2::SplitCandidate) = e1.improvement < e2.improvement
Base.isless(e1::SplitCandidate, e2::SplitCandidate) = e1.improvement < e2.improvement

"""
	Train greedy regression model
	Args:
		- maxdepth:		maximum depth of tree
		- tolerance:	minimum improvement to MSE required
		- minbucket:	minimum number of observations in a split to attempt a split
"""
# function trainGreedyModel(Y::Union{Vector, BitArray{1}}, missingdata::DataFrame;
function regress_greedy(Y::Union{Vector, BitArray{1}}, missingdata::DataFrame;
						  maxdepth::Int = 3,
						  tolerance::Float64 = 0.01,
						  minbucket::Int = 10,
						#   missingdata::DataFrame = data
						  )
	data = zeroimpute(missingdata)

	gm = initializeGreedyModel(Y, data, missingdata)

	maxdepth == 1 && return gm

	trainIndices = try findall(data[!, :Test] .== 0) catch; collect(1:nrow(data)) end
	heap = BinaryMaxHeap{SplitCandidate}([bestSplit(gm, Y, data, 1, trainIndices, minbucket, missingdata)])
	
	while !isempty(heap)
		leafToSplit = pop!(heap)
		if leafToSplit.improvement > tolerance #* length(trainIndices)
			split!(gm, leafToSplit.leaf, leafToSplit.feature, 
					# leafToSplit.leftIntercept, leafToSplit.leftCoeffs, 
					# leafToSplit.rightIntercept, leafToSplit.rightCoeffs,
					leafToSplit.leftModel, leafToSplit.rightModel,
					leafToSplit.leftFeatures, #in left
					gm.nodes[leafToSplit.leaf].featuresOut, #out left
					leafToSplit.rightFeatures, #in right
					sort(vcat(gm.nodes[leafToSplit.leaf].featuresOut, leafToSplit.feature)), #out right
					leafToSplit.rightError, leafToSplit.leftError)
			L = length(gm.nodes)
			if gm.nodes[L].depth < maxdepth # only add to heap if below max depth
				push!(heap, bestSplit(gm, Y, data, L-1, leafToSplit.leftPoints, minbucket, missingdata))
				push!(heap, bestSplit(gm, Y, data, L, leafToSplit.rightPoints, minbucket, missingdata))
			end
		end
	end
	return gm
end

"""
	Initialize the greedy model with a single node, which always predicts the mean
"""
function initializeGreedyModel(Y::Union{Vector, BitArray{1}}, data::DataFrame, missingdata::DataFrame)
	# features = [i for (i, name) in enumerate(setdiff(Symbol.(names(data)), [:Test, :Id]))
	# 			if !any(ismissing.(data[!, name]))]
	# points = try findall(data[!, :Test] .== 0) catch ; collect(1:nrow(data)) end
	# intercept, coeffs, SSE = regressionCoefficients(Y, data, points, features)
	# root = LeafNode(1, 1, intercept, coeffs, features, Int[], SSE, 0)

	features = [i for i in names(data) if i ∉ ["Test", "Id"] && !any(ismissing.(missingdata[:, i]))]
	# features = [i for i in names(data) if i ∉ ["Test", "Id"] && !any(ismissing.(data[:, i]))]
	points = try findall(data[!, :Test] .== 0) catch ; collect(1:nrow(data)) end
	model, SSE = regressionCoefficients(Y, data, points)
	root = LeafNode(1, 1, model, features, [], SSE, 0)

	return GreedyModel([root], Set([1]), false)
end
# function initializeGreedyModel(Y::BitArray{1}, data::DataFrame)
# 	features = [i for (i, name) in enumerate(setdiff(Symbol.(names(data)), [:Test, :Id]))
# 				if !any(ismissing.(data[!, name]))]
# 	points = findall(data[!, :Test] .== 0)
# 	intercept, coeffs, LL = regressionCoefficients(Y, data, points, features)
# 	root = LeafNode(1, 1, intercept, coeffs, features, Int[], LL, 0)
# 	return GreedyModel([root], Set([1]), true)
# end

"""
	Find best split at a particular node
"""
function bestSplit(gm::GreedyModel, Y::Union{Vector, BitArray{1}}, data::DataFrame, node::Int,
				   points::Vector{Int}, minbucket::Int,
				   missingdata::DataFrame = data
				   )

	currentNode = gm.nodes[node]
	predictive_features = setdiff(setdiff(names(data), ["Id", "Test"]), currentNode.featuresOut)
	splittable_features = setdiff(predictive_features, currentNode.featuresIn)

	# p = Base.size(data, 2) - 2
	# X = Matrix(data[points, Not([:Test, :Id])])
	# y = Y[points]
	# X = Float64.(X[:, currentNode.featuresIn])
	# @show intersect(currentNode.featuresIn, currentNode.featuresOut)
	# @show setdiff(currentNode.featuresIn, features)
	# pred = predict(data[points,:], currentNode.model)
	# currentNode.intercept .+ X * currentNode.coeffs[currentNode.featuresIn]
	currentLoss = currentNode.currentError #gm.logistic ? logloss(y, sigmoid.(pred)) : sum((y .- pred) .^ 2)
	bestLoss = currentLoss, currentLoss, 0.
	bestFeature = names(data)[1]
	# bestCoeffs = (0., zeros(p), 0., zeros(p))
	bestModels = (currentNode.model,[])
	# for j = findall([any(ismissing.(missingdata[:,j])) for j in 1:Base.size(missingdata,2)]) #Search only through the features that can be missing
	# for j = findall([any(ismissing.(missingdata[:,j])) for j in 1:Base.size(missingdata,2)]) #Search only through the features that can be missing
	for j in [n for n in names(data[points,:]) if any(ismissing.(missingdata[points,n])) && n ∈ splittable_features]
		# if j in features
		# 	continue
		# end

		# featuresLeft = sort(collect(Set(vcat(features, j)))) #Adding j to list of never missing features.
		# featuresLeft = sort(vcat(features, j))
		pointsLeft = points[.!ismissing.(missingdata[points, j])]
		length(pointsLeft) < minbucket && continue
		
		# featuresRight = features
		# featuresRight = sort(setdiff(features, [j]))
		pointsRight = points[ismissing.(missingdata[points, j])]
		length(pointsRight) < minbucket && continue

		modelLeft, lossLeft = regressionCoefficients(Y, data, pointsLeft, features=predictive_features)
		modelRight, lossRight = regressionCoefficients(Y, data, pointsRight, features=setdiff(predictive_features,[j]))

		# intLeft, coeffsLeft, lossLeft = regressionCoefficients(Y, data, pointsLeft, featuresLeft)
		# intRight, coeffsRight, lossRight = regressionCoefficients(Y, data, pointsRight, featuresRight)

		newLoss = (lossLeft + lossRight)
		if newLoss < bestLoss[1]
			bestLoss = newLoss, lossLeft, lossRight
			bestFeature = j
			# bestCoeffs = intLeft, coeffsLeft, intRight, coeffsRight
			bestModels = modelLeft, modelRight
		end
	end
	#REMARK: If no split possible with at least minbucket on each side, function will return the initial split with improvement 0% --> not implemented
	return SplitCandidate(node, (currentLoss - bestLoss[1])/abs(currentLoss), 
							bestFeature,
	                      	points[.!ismissing.(missingdata[points, bestFeature])],
	                      	points[ismissing.(missingdata[points, bestFeature])],
	                      	sort(collect(Set(vcat(currentNode.featuresIn, bestFeature)))), 
							sort(currentNode.featuresIn), # features, #sort(setdiff(features,[bestFeature])),
	                    #   bestCoeffs[1], bestCoeffs[2], bestCoeffs[3], bestCoeffs[4],
							bestModels[1], bestModels[2],
						  	bestLoss[2], bestLoss[3])
end

"""
	Execute a split of a particular leaf
"""
function split!(gm::GreedyModel, node::Int, feature::String,
				# intLeft::Float64, coeffsLeft::Vector{Float64},
				# intRight::Float64, coeffsRight::Vector{Float64},
				modelLeft, modelRight,
				featuresInLeft::Vector{String}, featuresOutLeft::Vector{String},
				featuresInRight::Vector{String}, featuresOutRight::Vector{String},
				errorLeft::Float64, errorRight::Float64)
	parentNode = gm.nodes[node]
	# leftNode = LeafNode(length(gm.nodes) + 1, parentNode.depth + 1,
	#                     intLeft, coeffsLeft, featuresInLeft, featuresOutLeft, errorLeft, parentNode.id)
	# rightNode = LeafNode(length(gm.nodes) + 2, parentNode.depth + 1,
	#                      intRight, coeffsRight, featuresInRight, featuresOutRight, errorRight, parentNode.id)
	# parentNode = SplitNode(parentNode.id, parentNode.depth, feature,
	#                        length(gm.nodes) + 1, length(gm.nodes) + 2,
	#                        parentNode.intercept, parentNode.coeffs, parentNode.currentError, parentNode.parent)
	leftNode = LeafNode(length(gm.nodes) + 1, parentNode.depth + 1,
								modelLeft, featuresInLeft, featuresOutLeft, errorLeft, parentNode.id)
	rightNode = LeafNode(length(gm.nodes) + 2, parentNode.depth + 1,
								modelRight, featuresInRight, featuresOutRight, errorRight, parentNode.id)
	parentNode = SplitNode(parentNode.id, parentNode.depth, feature,
	                       length(gm.nodes) + 1, length(gm.nodes) + 2,
	                       parentNode.model, parentNode.currentError, parentNode.parent)
	gm.nodes[node] = parentNode
	push!(gm.leaves, length(gm.nodes) + 1)
	push!(gm.leaves, length(gm.nodes) + 2)
	push!(gm.nodes, leftNode)
	push!(gm.nodes, rightNode)
	delete!(gm.leaves, parentNode.id)
end

"""
	Regression subroutine
	Args:
		- Y:`				the dependent variable data
		- data:				the data (X)
		- points: 			the points that will actually be in the leaf
		- features:			the features we're allowed to use
"""
function regressionCoefficients(Y::Vector, data::DataFrame, points::Vector{Int}; features::Vector{String}=setdiff(names(data),["Id", "Test"]))
	# p = Base.size(data, 2) - 1
	# coeffs = zeros(p)
	# β_0 = mean(Y[points])
	# if length(features) == 0
	# 	return β_0, coeffs, sum((β_0 .- Y[points]) .^ 2)
	# end
	# X = Matrix(data[points, Not([:Test, :Id])])[:, features]
	# X = Float64.(X)
	# y = Y[points]
	# cv = glmnetcv(X, y)
	# if length(cv.meanloss) > 0
	# 	β_0 = cv.path.a0[argmin(cv.meanloss)]
	# 	coeffs[features] = cv.path.betas[:, argmin(cv.meanloss)]
	# end
	model = regress_linear(Y[points], data[points, features]; regtype=:missing_weight, alpha=0.1)
	# @show evaluate(Y[points], data[points,:], model)
	SSE = sum((predict(data[points,:], model) .- Y[points]).^2)
	# @show evaluate(Ymodel)
	# SSE = argmin(cv.meanloss)*Base.size(X,1) #Log loss corresponds to out-of-sample predictive power
	# return β_0, coeffs, SSE
	return model, SSE
end
function regressionCoefficients(Y::BitArray{1}, data::DataFrame, points::Vector{Int}; features::Vector{String}=setdiff(names(data),["Id", "Test"]))
	# p = Base.size(data, 2) - 1
	# coeffs = zeros(p)
	# β_0 = safelog(mean(Y[points]) / (1 - mean(Y[points])))
	# # if var(Y) .<= 1e10
	# # 	Y0 = round(Int, mean(Y))
	# # 	β_0 = 100*(2*Y0-1)
	# # 	@show Y0, β_0
	# # 	return β_0, coeffs, logloss(Y[points], Y0)
	# # end
	# if var(Y) .<= 1e-10
	# 	# return β_0, coeffs, logloss(Y[points], mean(Y[points]))
	# 	return β_0, coeffs, 0.5
	# end
	# if length(features) == 0
	# 	# return β_0, coeffs, logloss(Y[points], mean(Y[points]))
	# 	return β_0, coeffs, 0.5
	# end
	# X = Matrix(data[points, Not([:Test, :Id])])[:, features]
	# X = Float64.(X)
	# y = Y[points]
	# cv = glmnetcv(X, hcat(Float64.(.!y), Float64.(y)), GLMNet.Binomial())
	# if length(cv.meanloss) > 0
	# 	β_0 = cv.path.a0[argmin(cv.meanloss)]
	# 	coeffs[features] = cv.path.betas[:, argmin(cv.meanloss)]
	# end

	
	# pred = sigmoid.(β_0 .+ X * coeffs[features])
	# # LL = logloss(y, pred)
	# # LL = argmin(cv.meanloss)*Base.size(X,1)
	# LL = 1 - auc(y, pred)
	# return β_0, coeffs, LL

	model = regress_linear(Y[points], data[points, features]; regtype=:missing_weight, alpha=0.1)
	pred = predict(data[points,:], model)
	
	SSE = 1 - auc(Y[points], pred)
	return model, SSE
end

"""
	Apply greedy regression model to data with missing values
"""
function predict(missingdata::DataFrame, gm::GreedyModel)#; missingdata::DataFrame = data)
	data = zeroimpute(missingdata)
	truenames = setdiff(names(data), ["Test", "Id"])
	result = zeros(nrow(data))
	# loop through points
	for i = 1:nrow(data)
		currentNode = gm.nodes[1]
		while !(currentNode.id in gm.leaves)
			if !ismissing(missingdata[i, currentNode.feature])
				currentNode = gm.nodes[currentNode.left]
			else
				currentNode = gm.nodes[currentNode.right]
			end
		end
		# result[i] = currentNode.intercept
		# for f in currentNode.featuresIn
		# 	result[i] += currentNode.coeffs[f] * data[i, f]
		# end
		result[i] = predict(data[[i], intersect(truenames, names(currentNode.model))], currentNode.model)[1]
	end
	return result
	# if gm.logistic
	# 	return sigmoid.(result)
	# else
	# 	return result
	# end
end

"""
	Get presence vector (+1, -1 or 0 for features)
"""
function getPresenceVector(gm::GreedyModel, leaf::Int)	
	# pattern = 
	# 	zeros(Int, ncol(gm.nodes[end].model))
	currentNode = gm.nodes[leaf]
	# pattern[currentNode.featuresIn] .= 1
	# pattern[currentNode.featuresOut] .= -1
	# return pattern
	hcat(
	DataFrame(ones(length(currentNode.featuresIn))', currentNode.featuresIn),
	DataFrame(-ones(length(currentNode.featuresOut))', currentNode.featuresOut)
	)
	# 	pattern = 
	# 	zeros(Int, ncol(gm.nodes[end].model))
	# currentNode = gm.nodes[leaf]
	# pattern[currentNode.featuresIn] .= 1
	# pattern[currentNode.featuresOut] .= -1

	# return pattern

end

"""
	Assign an order to presence vectors
"""
function comp(pattern1::Vector{Int}, pattern2::Vector{Int})
	if sum(pattern1) == sum(pattern2)
		return isless(pattern1, pattern2)
	else
		return sum(pattern1) < sum(pattern2)
	end
end
# function comp(pattern1::DataFrame, pattern2::DataFrame, featurelist)
# 	for n in featurelist
# 		if n ∉ names(pattern1)
# 			pattern1[!,n] .= -1
# 		end
# 		if n ∉ names(pattern2)
# 			pattern2[!,n] .= -1
# 		end
# 	end
# 	sort!(featurelist)
# 	pmat = Matrix{Int}(vcat(pattern1[featurelist], pattern2[featurelist]))
# 	comp(pmat[1,:], pmat[2,:])
# end
"""
	Nice ASCII output for greedy RMD model
"""
function print_node(gm::GreedyModel, node::LeafNode, indent)
	print(indent, " predict ")
	# if eltype(d[pattern]) <: DataFrame
	model = node.model
	@printf("Y=%5.2f ", model[1,"Offset"])
	for feature in sort(names(model))
		if feature ∉ ["Offset", "Test", "Id"] && abs(model[1,feature]) > 0
			if model[1,feature] > 0
				print("+")
			else
				print("-")
			end
			@printf("%5.2f ", abs(model[1,feature]))
			print(" ", feature, " ")
		end
	end
	print("\n")
end
function print_node(gm::GreedyModel, node::SplitNode, indent)
	println(indent, " If ", node.feature, " is not missing")
	print_node(gm, gm.nodes[node.left], "    "*indent)
	
	println(indent, " Else if ", node.feature, " is missing")
	print_node(gm, gm.nodes[node.right], "    "*indent)
end
function print_ascii(gm::GreedyModel)
	print_node(gm, gm.nodes[1], "")
end
# function print_ascii(gm::GreedyModel)
# 	d = Dict()
# 	features = [] #Find all features
# 	for leaf in gm.leaves
# 		features = union(features, gm.nodes[leaf].featuresIn,  gm.nodes[leaf].featuresOut)
# 	end
# 	alwaysIn = deepcopy(features)
# 	for leaf in gm.leaves
# 		alwaysIn = intersect(alwaysIn, gm.nodes[leaf].featuresIn)
# 	end
# 	features = setdiff(features,alwaysIn)

# 	for
# 	sort!(features)

# 	@show length(features)

	
# 	patterns = DataFrame([ [] for _ in features], features)
# 	for leaf in gm.leaves
# 		pattern = getPresenceVector(gm, leaf)
# 		patterns = vcat(patterns, pattern[:,features])
# 	end
# 	@show patterns

# 	function comp(pattern1::DataFrame, pattern2::DataFrame)
# 		for n in features
# 			if n ∉ names(pattern1)
# 				pattern1[!,n] .= 0
# 			end
# 			if n ∉ names(pattern2)
# 				pattern2[!,n] .= 0
# 			end
# 		end
		
# 		pmat = Matrix{Int}(vcat(pattern1[:,features], pattern2[:,features]))
# 		if sum(pmat[1,:]) == sum(pmat[2,:])
# 			return isless(pmat[1,:], pmat[2,:])
# 		else
# 			return sum(pmat[1,:]) < sum(pmat[2,:])
# 		end
# 	end

# 	patternList = sort(collect(keys(d)), lt=comp, rev=false)
# 	@show keys(d)
# 	# patternList = collect(keys(d))
# 	# features = abs.(patternList[1]) .> 0
# 	# for pattern in patternList
# 	# 	features = features .| (abs.(pattern) .> 0)
# 	# end
# 	# features = sort(findall(features))

# 	# println("Features\t\tModel")
# 	# for feature in features
# 	# 	# @printf("%2d ", feature)
# 	# 	print(feature)
# 	# end
# 	# print("            ")
# 	# for (k, feature) in enumerate(features)
# 	# 	if k < 10
# 	# 		print(" ")
# 	# 	end
# 	# 	# @printf(" β%d   ", feature)
# 	# 	print(" β   ", feature)
# 	# end
# 	# print("\n")

# 	for pattern in patternList
# 		print("If missing: ")
# 		for feature in features
# 			if pattern[1,feature] > 0
# 				# print(" + ")
# 			elseif pattern[1,feature] < 0
# 				print(feature, "; ")
# 			else
# 				print("   ")
# 			end
# 		end
# 		print("\n")
# 		print("  Then ")
# 		# if eltype(d[pattern]) <: DataFrame
# 			coeffs = d[pattern]
# 			@printf("Y=%5.2f ", coeffs[1,"Offset"])
# 			for feature in features
# 				if pattern[1,feature] > 0
# 					if coeffs[1,feature] > 0
# 						print("+")
# 					else
# 						print("-")
# 					end
# 					@printf("%5.2f ", abs(coeffs[1,feature]))
# 					print(" ", feature, " ")
# 				else
# 					print("       ")
# 				end
# 			end
# 		# end
# 		print("\n")
# 	end
# 	return nothing
# end

# """
# 	Evaluate the fit quality of a greedy model on a dataset
# """
# function evaluate(Y::Array{Float64}, df::DataFrame, model::GreedyModel, missingdata::DataFrame=df)
# 	if model.logistic
# 		error("Cannot evaluate a logistic model on continuous labels")
# 	else
# 		trainmean = Statistics.mean(Y[df[:,:Test] .== 0])
# 		SST = sum((Y[df[:,:Test] .== 0] .- trainmean) .^ 2)
# 		OSSST = sum((Y[df[:,:Test] .== 1] .- trainmean) .^ 2)

# 		prediction = predict(df, model, missingdata=missingdata)
# 		R2 = 1 - sum((Y[df[:,:Test] .== 0] .- prediction[df[:,:Test] .== 0]) .^ 2)/SST
# 		OSR2 = 1 - sum((Y[df[:,:Test] .== 1] .- prediction[df[:,:Test] .== 1]) .^ 2)/OSSST
# 		return R2, OSR2
# 	end
# end
# function evaluate(Y::BitArray{1}, df::DataFrame, model::GreedyModel, missingdata::DataFrame=df;
# 				  metric::AbstractString = "auc")
# 	if model.logistic
# 		prediction = predict(df, model, missingdata=missingdata)
# 		if metric == "logloss"
# 			ll = logloss(Y[df[:,:Test] .== 0], prediction[df[:,:Test] .== 0])
# 			osll = logloss(Y[df[:,:Test] .== 1], prediction[df[:,:Test] .== 1])
# 			return ll, osll
# 		elseif metric == "auc"
# 			return auc(Y[df[:,:Test] .== 0], prediction[df[:,:Test] .== 0]),
# 				   auc(Y[df[:,:Test] .== 1], prediction[df[:,:Test] .== 1])
# 		else
# 			error("Unknown metric: $metric (only supports 'logloss', 'auc')")
# 		end
# 	else
# 		error("Cannot evaluate a linear model on binary labels")
# 	end
# end
