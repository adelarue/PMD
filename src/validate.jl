###################################
### validate.jl
### Validate hyperparameters in training
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Same as regress from regress.jl, but validates hyperparameters
	Args:
		- a vector of dependent variables
		- a dataframe with no missing values
	Parameters: same as regress, but as vectors, and the best combination will be chosen
	Returns:
		- a dataframe with a single row, with the same column names as the data
		except Test, and an additional Offset containing the constant term
"""
function regress_cv(Y::Vector, data::DataFrame;
					val_fraction::Real=0.2,
					regtype::Vector{Symbol}=[:none],
					lasso::Vector{Bool}=[false],
					alpha::Vector{Float64}=[0.8],
					missing_penalty::Vector{Float64}=[1.0])	
	if length(regtype) < length(lasso) || (length(regtype) == 1 && regtype[1] == :none)#If lasso provides more options or regtype nothing particular
		regtype = map(t -> t ? :lasso : :none, lasso)
	end

	# isolate training set
	newY = Y[data[!, :Test] .== 0]
	newdata = filter(row -> row[:Test] == 0, data)
	# designate some of training as testing/validation
	# val_indices = shuffle(1:nrow(newdata))[1:Int(floor(val_fraction * nrow(newdata)))]
	val_indices = findall(split_dataset(newdata; test_fraction = val_fraction, random=true))
	newdata[val_indices, :Test] .= 1
	bestmodel = regress(newY, newdata, regtype = regtype[1], alpha = alpha[1],
	                    missing_penalty = missing_penalty[1])
	bestOSR2 = evaluate(newY, newdata, bestmodel)[2]
	bestparams = (regtype[1], alpha[1], missing_penalty[1])
	for l in regtype, a in alpha, mp in missing_penalty
		newmodel = regress(newY, newdata, regtype = l, alpha = a, missing_penalty = mp)
		newOSR2 = evaluate(newY, newdata, newmodel)[2]
		if newOSR2 > bestOSR2
			bestOSR2 = newOSR2
			bestparams = (l, a, mp)
		end
	end
	# train model on full dataset using best parameters
	bestmodel = regress(Y, data, regtype = bestparams[1], alpha = bestparams[2],
	                    missing_penalty = bestparams[3])
	return bestmodel, bestparams
end
function regress_cv(Y::BitArray{1}, data::DataFrame;
					val_fraction::Real=0.2,
					lasso::Vector{Bool}=[false],
					regtype::Vector{Symbol}=[:none],
					alpha::Vector{Float64}=[0.8],
					missing_penalty::Vector{Float64}=[1.0])
	if length(regtype) < length(lasso) || (length(regtype) == 1 && regtype[1] == :none)#If lasso provides more options or regtype nothing particular
		regtype = map(t -> t ? :lasso : :none, lasso)
	end

	# isolate training set
	newY = Y[data[:, :Test] .== 0]
	newdata = filter(row -> row[:Test] == 0, data)
	# designate some of training as testing/validation
	val_indices = shuffle(1:nrow(newdata))[1:Int(floor(val_fraction * nrow(newdata)))]
	newdata[val_indices, :Test] .= 1
	@show regtype[1]
	bestmodel = regress(newY, newdata, regtype = regtype[1], alpha = alpha[1],
	                    missing_penalty = missing_penalty[1])
	bestlogloss = evaluate(newY, newdata, bestmodel)[2]
	bestparams = (regtype[1], alpha[1], missing_penalty[1])
	for l in regtype, a in alpha, mp in missing_penalty
		newmodel = regress(newY, newdata, regtype = l, alpha = a, missing_penalty = mp)
		newlogloss = evaluate(newY, newdata, newmodel)[2]
		if newlogloss < bestlogloss
			bestlogloss = newlogloss
			bestparams = (l, a, mp)
		end
	end
	# train model on full dataset using best parameters
	bestmodel = regress(Y, data, regtype = bestparams[1], alpha = bestparams[2],
	                    missing_penalty = bestparams[3])
	return bestmodel, bestparams
end



"""
	Train greedy regression model, validating hyperparameters
	Args:
		- maxdepth:		maximum depth of tree
		- tolerance:	minimum improvement to MSE required
		- minbucket:	minimum number of observations in a split to attempt a split
"""
function greedymodel_cv(Y::Union{Vector, BitArray{1}}, data::DataFrame;
						val_fraction::Real=0.2,
						maxdepth::Vector{Int} = [3],
						tolerance::Vector{Float64} = [0.1],
						minbucket::Vector{Int} = [10],
						missingdata::DataFrame = data)
	# isolate training set
	newY = Y[data[!, :Test] .== 0]
	newdata = filter(row -> row[:Test] == 0, data)
	newmissingdata = filter(row -> row[:Test] == 0, data)
	# designate some of training as testing/validation
	val_indices = shuffle(1:nrow(newdata))[1:Int(floor(val_fraction * nrow(newdata)))]
	newdata[val_indices, :Test] .= 1
	newmissingdata[val_indices, :Test] .= 1
	bestmodel = trainGreedyModel(newY, newdata, maxdepth = maxdepth[1],
	                             tolerance = tolerance[1],
	                    		 minbucket = minbucket[1],
	                    		 missingdata = newmissingdata)
	bestMetric = evaluate(newY, newdata, bestmodel, newmissingdata)[2]
	bestparams = (maxdepth[1], tolerance[1], minbucket[1])
	for depth in maxdepth, tol in tolerance, mb in minbucket
		if depth == maxdepth[1] && tol == tolerance[1] && mb == minbucket[1]
			continue
		end
		newmodel = trainGreedyModel(newY, newdata, maxdepth = depth, tolerance = tol,
	                    		 	 minbucket = mb, missingdata = newmissingdata)
		newMetric = evaluate(newY, newdata, newmodel, newmissingdata)[2]
		if (!newmodel.logistic && newMetric > bestMetric) || (newmodel.logistic && newMetric < bestMetric)
			bestMetric = newMetric
			bestparams = (depth, tol, mb)
		end
	end
	# train model on full dataset using best parameters
	bestmodel = trainGreedyModel(Y, data, maxdepth = bestparams[1],
	                             tolerance = bestparams[2],
	                    		 minbucket = bestparams[3],
	                    		 missingdata = missingdata)
	return bestmodel, bestparams
end
