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

##Wrapper for all the regress_XXX methods
function regress(Y, data::DataFrame; 
		model::Symbol=:linear,
		parameter_dict::Dict{Symbol,T}=Dict(:regtype => :missing_weight)) where T	
	if model == :linear 
		regress_linear(Y, data; parameter_dict...)
	elseif model == :tree || model == :friedman
		regress_tree(Y, data; parameter_dict...)
	elseif model == :rf
		regress_rf(Y, data; parameter_dict...)
	elseif model == :nn
		regress_nn(Y, data; parameter_dict...)
	elseif model == :greedy
		regress_greedy(Y, data; parameter_dict...)
	elseif model == :joint
		impute_then_regress(Y, data; parameter_dict=parameter_dict)
	end
end

##Quick tool to expand a dictionary of lists into a list of dictionaries
function expand(d::Dict)
	r = []
	sizes = [length(d[k]) for k in keys(d)]	
	for i in product(sizes)
		push!(r, Dict([k => d[k][round(Int,i[kk])] for (kk,k) in enumerate(keys(d))]))
	end
	return r
end
function product(l::Vector{Int64}, idx, n)    
    if length(l) == 1
        @assert idx == n
        r = []
        for i in 1:l[1]
            a = zeros(n); a[idx] = i
            push!(r, a)
        end
        return r
    elseif length(l) > 1
        res = []
        for i in 1:l[1]
            r = product(l[2:end], idx+1, n)
            for t in r 
                t[idx] = i 
                push!(res, t)
            end
        end
        return res
    end
end

product(l::Vector{Int64}) = product(l,1,length(l))



function regress_cv(Y, data::DataFrame;
					val_fraction::Real=0.2,
					model::Symbol=:linear,
					parameter_dict::Dict{Symbol,T}=Dict()) where T

	@assert length(parameter_dict) > 0
	
	# For each supported type of predictive model, checks that parameter_dict corresponds to valid hyper-parameters 
	if model == :linear 
		keys(parameter_dict) ⊆ [:regtype, :alpha, :missing_penalty]
	elseif model == :tree || model == :friedman
		keys(parameter_dict) ⊆ [:maxdepth]
	elseif model == :rf
		keys(parameter_dict) ⊆ [:maxdepth, :nfeat, :ntrees, :psamples]
	elseif model == :nn 
		keys(parameter_dict) ⊆ [:hidden_nodes]
	elseif model == :greedy
		keys(parameter_dict) ⊆ [:maxdepth, :tolerance, :minbucket]
	# elseif model == :joint
	# 	keys(parameter_dict) ⊆ [:modeltype, :parameter_dict]
	end

	# Isolate training set
	newY = Y[data[!, :Test] .== 0]
	newdata = filter(row -> row[:Test] == 0, data)
	
	# Designate some of training as testing/validation
	val_indices = findall(split_dataset(newdata; test_fraction = val_fraction, random=true))
	newdata[val_indices, :Test] .= 1

	# bestmodel = []
	bestOSR2 = -Inf
	bestparams = [] 

	for params in expand(parameter_dict)
		newmodel = regress(newY, newdata; model=model, parameter_dict=params)
		# @show newmodel
		# @show params, evaluate(newY, newdata, newmodel)
		newOSR2 = evaluate(newY, newdata, newmodel)[2]
		if newOSR2 > bestOSR2
			bestOSR2 = newOSR2
			bestparams = params
		end
	end
	# println("Found best hyper-parameters. Computing full model")
	# train model on full dataset using best parameters
	bestmodel = regress(newY, newdata; model=model, parameter_dict=bestparams)

	return bestmodel, bestparams
end