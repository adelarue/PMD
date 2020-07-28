###################################################
## greedy.jl
##      Greedy tree-like method for regression
## Author: Arthur Delarue, Jean Pauphilet, 2019
###################################################

import Base.<

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
	feature::Int
	"Left child (feature present)"
	left::Int
	"Right child (feature absent)"
	right::Int
	"Constant term in regression model"
	intercept::Float64
	"Regression coefficients"
	coeffs::Vector{Float64}
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
	"Constant term in regression model"
	intercept::Float64
	"Regression coefficients"
	coeffs::Vector{Float64}
	"Features included"
	featuresIn::Vector{Int}
	"Features excluded"
	featuresOut::Vector{Int}
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
	feature::Int
	"Set of new left points"
	leftPoints::Vector{Int}
	"Set of new right points"
	rightPoints::Vector{Int}
	"Set of new left features"
	leftFeatures::Vector{Int}
	"Set of new right features"
	rightFeatures::Vector{Int}
	"New left intercept"
	leftIntercept::Float64
	"New left coefficients"
	leftCoeffs::Vector{Float64}
	"New right intercept"
	rightIntercept::Float64
	"New right coeffs"
	rightCoeffs::Vector{Float64}
end
<(e1::SplitCandidate, e2::SplitCandidate) = e1.improvement < e2.improvement

"""
	Train greedy regression model
	Args:
		- maxdepth:		maximum depth of tree
		- tolerance:	minimum improvement to MSE required
		- minbucket:	minimum number of observations in a split to attempt a split
"""
function trainGreedyModel(Y::Vector, data::DataFrame;
						  maxdepth::Int = 3,
						  tolerance::Float64 = 0.1,
						  minbucket::Int = 10,
						  missingdata::DataFrame = data)
	gm = initializeGreedyModel(Y, data)
	maxdepth == 1 && return gm
	trainIndices = findall(data[!, :Test] .== 0)
	heap = BinaryMaxHeap([bestSplit(gm, Y, data, 1, trainIndices,
	                                gm.nodes[1].featuresIn, minbucket, missingdata)])
	while !isempty(heap)
		leafToSplit = pop!(heap)
		if leafToSplit.improvement > tolerance * length(trainIndices)
			split!(gm, leafToSplit.leaf, leafToSplit.feature, leafToSplit.leftIntercept,
			       leafToSplit.leftCoeffs, leafToSplit.rightIntercept, leafToSplit.rightCoeffs,
			       leafToSplit.leftFeatures, gm.nodes[leafToSplit.leaf].featuresOut,
			       leafToSplit.rightFeatures,
			       sort(vcat(gm.nodes[leafToSplit.leaf].featuresOut, leafToSplit.feature)))
			L = length(gm.nodes)
			if gm.nodes[L].depth < maxdepth # only add to heap if below max depth
				push!(heap, bestSplit(gm, Y, data, L-1, leafToSplit.leftPoints,
				                      leafToSplit.leftFeatures, minbucket, missingdata))
				push!(heap, bestSplit(gm, Y, data, L, leafToSplit.rightPoints,
				                      leafToSplit.rightFeatures, minbucket, missingdata))
			end
		end
	end
	return gm
end

"""
	Initialize the greedy model with a single node, which always predicts the mean
"""
function initializeGreedyModel(Y::Vector, data::DataFrame)
	features = [i for (i, name) in enumerate(setdiff(Symbol.(names(data)), [:Test]))
				if !any(ismissing.(data[!, name]))]
	points = findall(data[!, :Test] .== 0)
	intercept, coeffs, SSE = regressionCoefficients(Y, data, points, features)
	root = LeafNode(1, 1, intercept, coeffs, features, Int[], 0)
	return GreedyModel([root], Set([1]))
end

"""
	Find best split at a particular node
"""
function bestSplit(gm::GreedyModel, Y::Vector, data::DataFrame, node::Int,
				   points::Vector{Int}, features::Vector{Int}, minbucket::Int,
				   missingdata::DataFrame = data)
	currentNode = gm.nodes[node]
	p = Base.size(data, 2) - 1
	X = Matrix(data[points, Not(:Test)])
	y = Y[points]
	X = Float64.(X[:, currentNode.featuresIn])
	currentSSE = sum((currentNode.intercept .+
	                  X * currentNode.coeffs[currentNode.featuresIn] - y) .^ 2)
	bestSSE = currentSSE
	bestFeature = 1
	bestCoeffs = (0., zeros(p), 0., zeros(p))
	for j = 1:p
		# if j in features
		# 	continue
		# end
		featuresLeft = sort(collect(Set(vcat(features, j))))
		# featuresLeft = sort(vcat(features, j))
		pointsLeft = points[.!ismissing.(missingdata[points, j])]
		length(pointsLeft) < minbucket && continue
		intLeft, coeffsLeft, SSEL = regressionCoefficients(Y, data, pointsLeft, featuresLeft)
		featuresRight = features
		pointsRight = points[ismissing.(missingdata[points, j])]
		length(pointsRight) < minbucket && continue
		intRight, coeffsRight, SSER = regressionCoefficients(Y, data, pointsRight, featuresRight)
		newSSE = SSEL + SSER
		if newSSE < bestSSE
			bestSSE = newSSE
			bestFeature = j
			bestCoeffs = intLeft, coeffsLeft, intRight, coeffsRight
		end
	end
	return SplitCandidate(node, abs(bestSSE - currentSSE), bestFeature,
	                      points[.!ismissing.(missingdata[points, bestFeature])],
	                      points[ismissing.(missingdata[points, bestFeature])],
	                      sort(collect(Set(vcat(features, bestFeature)))), features,
	                      bestCoeffs[1], bestCoeffs[2], bestCoeffs[3], bestCoeffs[4])
end

"""
	Execute a split of a particular leaf
"""
function split!(gm::GreedyModel, node::Int, feature::Int,
				intLeft::Float64, coeffsLeft::Vector{Float64},
				intRight::Float64, coeffsRight::Vector{Float64},
				featuresInLeft::Vector{Int}, featuresOutLeft::Vector{Int},
				featuresInRight::Vector{Int}, featuresOutRight::Vector{Int})
	parentNode = gm.nodes[node]
	leftNode = LeafNode(length(gm.nodes) + 1, parentNode.depth + 1,
	                    intLeft, coeffsLeft, featuresInLeft, featuresOutLeft, parentNode.id)
	rightNode = LeafNode(length(gm.nodes) + 2, parentNode.depth + 1,
	                     intRight, coeffsRight, featuresInRight, featuresOutRight, parentNode.id)
	parentNode = SplitNode(parentNode.id, parentNode.depth, feature,
	                       length(gm.nodes) + 1, length(gm.nodes) + 2,
	                       parentNode.intercept, parentNode.coeffs, parentNode.parent)
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
function regressionCoefficients(Y::Vector, data::DataFrame, points::Vector{Int},
								features::Vector{Int})
	p = Base.size(data, 2) - 1
	coeffs = zeros(p)
	if length(features) == 0
		β_0 = mean(Y[points])
		return β_0, coeffs, sum((β_0 .- Y[points]) .^ 2)
	end
	X = Matrix(data[points, Not(:Test)])[:, features]
	X = Float64.(X)
	y = Y[points]
	cv = glmnetcv(X, y)
	β_0 = cv.path.a0[argmin(cv.meanloss)]
	coeffs[features] = cv.path.betas[:, argmin(cv.meanloss)]
	SSE = sum((X * coeffs[features] - y .+ β_0) .^ 2)
	return β_0, coeffs, SSE
end

"""
	Apply greedy regression model to data with missing values
"""
function predict(data::DataFrame, gm::GreedyModel; missingdata::DataFrame = data)
	truenames = setdiff(Symbol.(names(data)), [:Test])
	result = zeros(nrow(data))
	# loop through points
	for i = 1:nrow(data)
		currentNode = gm.nodes[1]
		while !(currentNode.id in gm.leaves)
			if !ismissing(missingdata[i, truenames[currentNode.feature]])
				currentNode = gm.nodes[currentNode.left]
			else
				currentNode = gm.nodes[currentNode.right]
			end
		end
		result[i] = currentNode.intercept
		for f in currentNode.featuresIn
			result[i] += currentNode.coeffs[f] * data[i, f]
		end
	end
	return result
end

"""
	Get presence vector (+1, -1 or 0 for features)
"""
function getPresenceVector(gm::GreedyModel, leaf::Int)
	pattern = zeros(Int, length(gm.nodes[end].coeffs))
	currentNode = gm.nodes[leaf]
	pattern[currentNode.featuresIn] .= 1
	pattern[currentNode.featuresOut] .= -1
	return pattern
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

"""
	Nice ASCII output for greedy RMD model
"""
function print_ascii(gm::GreedyModel)
	d = Dict()
	for leaf in gm.leaves
		pattern = getPresenceVector(gm, leaf)
		d[pattern] = (gm.nodes[leaf].intercept, gm.nodes[leaf].coeffs)
	end
	patternList = sort(collect(keys(d)), lt=comp, rev=true)
	features = abs.(patternList[1]) .> 0
	for pattern in patternList
		features = features .| (abs.(pattern) .> 0)
	end
	features = sort(findall(features))
	println("Features\t\tModel")
	for feature in features
		@printf("%2d ", feature)
	end
	print("            ")
	for feature in features
		if feature < 10
			print(" ")
		end
		@printf(" β%d   ", feature)
	end
	print("\n")
	for pattern in patternList
		for feature in features
			if pattern[feature] > 0
				print(" + ")
			elseif pattern[feature] < 0
				print(" - ")
			else
				print("   ")
			end
		end
		print("  ")
		@printf("Y=%5.2f ", d[pattern][1])
		coeffs = d[pattern][2]
		for feature in features
			if pattern[feature] > 0
				if coeffs[feature] > 0
					print("+")
				else
					print("-")
				end
				@printf("%5.2f ", abs(coeffs[feature]))
			else
				print("       ")
			end
		end
		print("\n")
	end
	return nothing
end

"""
	Evaluate the fit quality of a greedy model on a dataset
"""
function evaluate(Y::Array{Float64}, df::DataFrame, model::GreedyModel, missingdata::DataFrame=df)
	trainmean = Statistics.mean(Y[df[:,:Test] .== 0])
	SST = sum((Y[df[:,:Test] .== 0] .- trainmean) .^ 2)
	OSSST = sum((Y[df[:,:Test] .== 1] .- trainmean) .^ 2)

	prediction = predict(df, model, missingdata=missingdata)
	R2 = 1 - sum((Y[df[:,:Test] .== 0] .- prediction[df[:,:Test] .== 0]) .^ 2)/SST
	OSR2 = 1 - sum((Y[df[:,:Test] .== 1] .- prediction[df[:,:Test] .== 1]) .^ 2)/OSSST
	return R2, OSR2
end
