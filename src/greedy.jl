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
	"Features included"
	featuresIn::Vector{String}
	"Features excluded"
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

	gm = initializeGreedyModel(Y, data)

	maxdepth == 1 && return gm

	trainIndices = try findall(data[!, :Test] .== 0) catch; collect(1:nrow(data)) end
	heap = BinaryMaxHeap{SplitCandidate}([bestSplit(gm, Y, data, 1, trainIndices,
	                                gm.nodes[1].featuresIn, minbucket, missingdata)])
	
	while !isempty(heap)
		leafToSplit = pop!(heap)
		if leafToSplit.improvement > tolerance #* length(trainIndices)
			split!(gm, leafToSplit.leaf, leafToSplit.feature, 
					# leafToSplit.leftIntercept, leafToSplit.leftCoeffs, 
					# leafToSplit.rightIntercept, leafToSplit.rightCoeffs,
					leafToSplit.leftModel, leafToSplit.rightModel,
			       leafToSplit.leftFeatures, gm.nodes[leafToSplit.leaf].featuresOut,
			       leafToSplit.rightFeatures,
			       sort(vcat(gm.nodes[leafToSplit.leaf].featuresOut, leafToSplit.feature)),
				   leafToSplit.rightError, leafToSplit.leftError)
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
function initializeGreedyModel(Y::Union{Vector, BitArray{1}}, data::DataFrame)
	# features = [i for (i, name) in enumerate(setdiff(Symbol.(names(data)), [:Test, :Id]))
	# 			if !any(ismissing.(data[!, name]))]
	# points = try findall(data[!, :Test] .== 0) catch ; collect(1:nrow(data)) end
	# intercept, coeffs, SSE = regressionCoefficients(Y, data, points, features)
	# root = LeafNode(1, 1, intercept, coeffs, features, Int[], SSE, 0)

	features = [i for i in names(data) if i ∉ ["Test", "Id"] && !any(ismissing.(data[!, i]))]
	points = try findall(data[!, :Test] .== 0) catch ; collect(1:nrow(data)) end
	model, SSE = regressionCoefficients(Y, data, points, features)
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
				   points::Vector{Int}, features::Vector{String}, minbucket::Int,
				   missingdata::DataFrame = data
				   )

	currentNode = gm.nodes[node]
	# p = Base.size(data, 2) - 2
	# X = Matrix(data[points, Not([:Test, :Id])])
	# y = Y[points]
	# X = Float64.(X[:, currentNode.featuresIn])

	pred = predict(data[points,:], currentNode.model)
	# currentNode.intercept .+ X * currentNode.coeffs[currentNode.featuresIn]
	currentLoss = currentNode.currentError #gm.logistic ? logloss(y, sigmoid.(pred)) : sum((y .- pred) .^ 2)
	bestLoss = currentLoss, currentLoss, 0.
	bestFeature = names(data)[1]
	# bestCoeffs = (0., zeros(p), 0., zeros(p))
	bestModels = (currentNode.model,[])
	# for j = findall([any(ismissing.(missingdata[:,j])) for j in 1:Base.size(missingdata,2)]) #Search only through the features that can be missing
	# for j = findall([any(ismissing.(missingdata[:,j])) for j in 1:Base.size(missingdata,2)]) #Search only through the features that can be missing
	for j in [n for n in names(data[points,:]) if any(ismissing.(missingdata[points,n])) && n ∈ features]
		# if j in features
		# 	continue
		# end
		featuresLeft = sort(collect(Set(vcat(features, j)))) #Not needed since using zero-impute rather than complete features within each leaf
		# featuresLeft = sort(vcat(features, j))
		pointsLeft = points[.!ismissing.(missingdata[points, j])]
		length(pointsLeft) < minbucket && continue

		featuresRight = features
		pointsRight = points[ismissing.(missingdata[points, j])]
		length(pointsRight) < minbucket && continue

		modelLeft, lossLeft = regressionCoefficients(Y, data, pointsLeft, featuresLeft)
		modelRight, lossRight = regressionCoefficients(Y, data, pointsRight, featuresRight)

		# intLeft, coeffsLeft, lossLeft = regressionCoefficients(Y, data, pointsLeft, featuresLeft)
		# intRight, coeffsRight, lossRight = regressionCoefficients(Y, data, pointsRight, featuresRight)

		newLoss = (lossLeft + lossRight)
		if newLoss > bestLoss[1]
			bestLoss = newLoss, lossLeft, lossRight
			bestFeature = j
			# bestCoeffs = intLeft, coeffsLeft, intRight, coeffsRight
			bestModels = modelLeft, modelRight
		end
	end
	return SplitCandidate(node, (currentLoss - bestLoss[1])/abs(currentLoss), 
							bestFeature,
	                      	points[.!ismissing.(missingdata[points, bestFeature])],
	                      	points[ismissing.(missingdata[points, bestFeature])],
	                      	sort(collect(Set(vcat(features, bestFeature)))), features,
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
function regressionCoefficients(Y::Vector, data::DataFrame, points::Vector{Int}, features::Vector{String})
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
	model = regress_linear(Y, data[points, features]; regtype=:missing_weight, alpha=0.1)
	
	SSE = sum((predict(data[points,:], model) .- Y[points]).^2)
	# SSE = argmin(cv.meanloss)*Base.size(X,1) #Log loss corresponds to out-of-sample predictive power
	# return β_0, coeffs, SSE
	return model, SSE
end
function regressionCoefficients(Y::BitArray{1}, data::DataFrame, points::Vector{Int}, features::Vector{String})
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

	model = regress_linear(Y, data[points, features]; regtype=:missing_weight, alpha=0.1)
	pred = predict(data[points,:], model)
	SSE = 1 - auc(Y[points], pred)
	return model, SSE
end

"""
	Apply greedy regression model to data with missing values
"""
function predict(missingdata::DataFrame, gm::GreedyModel)#; missingdata::DataFrame = data)
	data = zeroimpute(missingdata)
	truenames = setdiff(Symbol.(names(data)), [:Test, :Id])
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
		result[i] = predict(data[[i], currentNode.featuresIn], currentNode.model)[1]
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
