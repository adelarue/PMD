###################################
### generate_y.jl
### Code to generate artificial dependent variable
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Standardize dataset
	Args:
		- data: a DataFrame
	Keyword args:
		- separate_test: whether to standardize using the mean and standard deviation of
			the training set only
	Returns:
		- a DataFrame that has been standardized (each column has mean zero
			and standard deviation one), except special columns
"""
function standardize(data::DataFrame)
	# @assert count_missing_columns(data) == 0
	truenames = setdiff(Symbol.(names(data)), [:Test, :Id, :Y])
	μ = [mean(data[.!ismissing.(data[:, name]), name]) for name in truenames]
	σ = [std(data[.!ismissing.(data[:, name]), name]) for name in truenames]
	# if a stddev is 0, the column is constant, and we should simply not rescale it
	σ[findall(σ .== 0)] .= 1
	newdata = DataFrame()
	for (i, name) in enumerate(truenames)
		newdata[!, name] = (data[!, name] .- μ[i]) ./ σ[i]
	end
	for n in intersect([:Test, :Id, :Y], Symbol.(names(data)))
		newdata[!,n] = data[!,n]
	end
	return newdata
end

# """
# 	Generate Y, linearly dependent on the features of X. More precisely, the true model has
# 		(possibly soft-thresholded) standard normal coefficients and a U([0, 1]) bias, to
# 		which we add zero-mean Gaussian noise
# 	Args:
# 		- data: DataFrame with fully observed values
# 	Keyword args:
# 		- soft_threshold: soft threshold subtracted in absolute value from all true coefficients
# 			(see softthresholding function)
# 		- SNR: signal-to-noise ratio
# 	Returns:
# 		- a vector of length nrow(data) with the values of Y
# """
# function linear_y(data::DataFrame, data_missing::DataFrame;
# 	k::Real=10, SNR::Real=4,
#     canbemissing=falses(Base.size(data,2)), #indicates which features can be missing
# 	mar::Bool=true,
#     k_missing_in_signal::Int=0) #indicates the number of potentially missing features in signal

#     @assert k >= 0.0
#     @assert SNR >= 0.0

# 	feature_names = Symbol.(names(data));
#     nevermissing_features = feature_names[.!canbemissing]; missing_features = feature_names[canbemissing]
# 	# @show length(missing_features)
# 	setdiff!(feature_names, [:Test, :Id]); setdiff!(nevermissing_features, [:Test, :Id]); setdiff!(missing_features, [:Test, :Id]);
# 	# @show length(missing_features)
# 	k = min(k, length(feature_names))
#     k_missing_in_signal = min(k_missing_in_signal, length(missing_features))
# 	# @show k_missing_in_signal
# 	k_non_missing = min(max(k - k_missing_in_signal, 0), length(nevermissing_features))

#     #Standardize
#     newdata = standardize(data[:,feature_names])

#     Y = zeros(nrow(newdata))
#     #For nevermissing features, choose then generate
# 	# @show k_non_missing
# 	if k_non_missing > 0
# 		support = shuffle(nevermissing_features)[1:k_non_missing]
# 		w1 = 2*rand(k_non_missing) .- 1
# 		Y += Matrix{Float64}(newdata[:,support])*w1
# 	end
#     #For missing feautres, choose
# 	# @show k_missing_in_signal
# 	if k_missing_in_signal > 0
# 	    support = shuffle(missing_features)[1:k_missing_in_signal]
# 	    w2 = 2*rand(k_missing_in_signal) .- 1
# 	    Y += Matrix{Float64}(newdata[:,support])*w2
# 		# if !mar
# 		# 	w2m = 2*rand(k_missing_in_signal) .- 1
# 		# 	Y += Matrix{Float64}(1.0 .* ismissing.(data_missing[:,support]) )*w2m
# 		# end
# 	end
# 	if k_missing_in_signal > 0 && !mar
# 		support = shuffle(missing_features)[1:k_missing_in_signal]
# 		w2m = 2*rand(k_missing_in_signal) .- 1
# 		Y += Matrix{Float64}(1.0 .* ismissing.(data_missing[:,support]) )*w2m
# 	end
#     #Add bias
#     btrue = randn(1); Y .+= btrue
#     #Add noise
#     noise = randn(nrow(newdata)); noise .*= norm(Y) / norm(noise) / SNR

#     return Y .+ noise, k_non_missing+k_missing_in_signal, k_missing_in_signal
# end

"Sigmoid function"
sigmoid(x::Real) = 1 / (1 + exp(-1 * x))

"Generate binary Y"
function binary_y(data::DataFrame, data_missing::DataFrame;
				  k::Real=10,  k_missing_in_signal::Int=0,  
				  SNR::Real=4,
    			  canbemissing=falses(Base.size(data,2)), #indicates which features can be missing
				  mar::Bool=true,
				  model::Symbol = :linear, hidden_nodes::Int=10,
    			  posfraction::Float64=0.5)

	Y, k1, k2 = generate_y(data, data_missing, k=k, k_missing_in_signal=k_missing_in_signal, canbemissing=canbemissing,
	                     mar=mar, SNR=SNR, model=model, hidden_nodes=hidden_nodes)
	vY = sigmoid.(Y)
	sigmoid_threshold = sort(vY, rev=true)[round(Int,posfraction*nrow(data))]		 
	return sigmoid.(Y) .> sigmoid_threshold, k1, k2
end


"""
	Soft thresholding
"""
function softthresholding(x::Real; λ::Real=0.1)
    if x > λ
        return x - λ
    elseif x < -λ
        return x + λ
    else
        return 0
    end
end

# """
# 	Generate Y, nonlinearly dependent on the features of X. More precisely, the true model
# 		is a neural network with a single hidden layer of N nodes, and ReLU activations,
# 		plus some Gaussian noise
# 	Args:
# 		- data: DataFrame with fully observed values
# 		- data_missing: DataFrame with missing values
# 		- k: number of features that can be missing
# 		- canbemissing: indicates which features can be missing
# 		- mar: whether the signal does not depend on missingness indicators
# 		- k_missing_in_signal: number of potentially missing features in signal
# 		- hidden_nodes: number of hidden nodes
# 	Returns:
# 		- a vector of length nrow(data) with the values of Y
# """
# function nonlinear_y(data::DataFrame,
# 	data_missing::DataFrame;
# 	k::Real=10, SNR::Real=4,
#     canbemissing=falses(Base.size(data,2)),
# 	mar::Bool=true,
#     k_missing_in_signal::Int=0,
#     hidden_nodes::Int=10) 

#     @assert k >= 0.0
#     @assert SNR >= 0.0

# 	feature_names = Symbol.(names(data));
#     nevermissing_features = feature_names[.!canbemissing]; missing_features = feature_names[canbemissing]
# 	setdiff!(feature_names, [:Test, :Id]); setdiff!(nevermissing_features, [:Test, :Id]); setdiff!(missing_features, [:Test, :Id]);

# 	k = min(k, length(feature_names))
#     k_missing_in_signal = min(k_missing_in_signal, length(missing_features))
# 	k_non_missing = min(max(k - k_missing_in_signal, 0), length(nevermissing_features))

#     #Standardize
#     newdata = standardize(data[:,feature_names])

#     # Start creating weight matrix as well as input matrix
#     # note that input data matrix needs to be transposed (columns are data points)
#     X = zeros(0, nrow(newdata))
#     W = zeros(hidden_nodes, 0)

#     #For nevermissing features, choose then generate
# 	if k_non_missing > 0
# 		support = shuffle(nevermissing_features)[1:k_non_missing]
# 		X = vcat(X, Matrix{Float64}(newdata[:,support])')
# 		W = hcat(W, 2*rand(hidden_nodes, k_non_missing) .- 1)
# 	end
#     #For missing feautres, choose
# 	# @show k_missing_in_signal
# 	if k_missing_in_signal > 0
# 	    support = shuffle(missing_features)[1:k_missing_in_signal]
# 	    X = vcat(X, Matrix{Float64}(newdata[:,support])')
# 		W = hcat(W, 2*rand(hidden_nodes, k_missing_in_signal) .- 1)
# 	end
# 	if k_missing_in_signal > 0 && !mar
# 		support = shuffle(missing_features)[1:k_missing_in_signal]
# 		W = hcat(W, 2*rand(hidden_nodes, k_missing_in_signal) .- 1)
# 		X = vcat(X, Matrix{Float64}(1.0 .* ismissing.(data_missing[:,support]))')
# 	end
#     #Add bias
#     nn = Flux.Chain(Flux.Dense(W, # linear weights
#                                randn(hidden_nodes), # bias of each hidden node
#                                Flux.relu), # activation function
#                     Flux.Dense(2 * rand(1, hidden_nodes) .- 1, # linear weights
#                                randn(1), # bias of each hidden node
#                                x->x)) # activation function of output layer (identity)
#     Y = vec(nn(X))

#     #Add noise
#     noise = randn(nrow(newdata)); noise .*= norm(Y) / norm(noise) / SNR

#     return Y .+ noise, k_non_missing+k_missing_in_signal, k_missing_in_signal
# end

"""
	Generate Y, nonlinearly, following Friedman (1991) model 
		Y = 10 sin(π X1 X2) + 20(X3 − 0.5)2 + 1 X4 + 5 X5 + ε
	Args:
		- data: DataFrame with fully observed values
		- data_missing: DataFrame with missing values
		- k: number of features that can be missing
		- canbemissing: indicates which features can be missing
		- mar: whether the signal does not depend on missingness indicators
		- k_missing_in_signal: number of potentially missing features in signal
		- hidden_nodes: number of hidden nodes
	Returns:
		- a vector of length nrow(data) with the values of Y
"""
function generate_y(data::DataFrame,
	data_missing::DataFrame;
	k::Real=10, k_missing_in_signal::Int=0, canbemissing=falses(Base.size(data,2)),
	mar::Bool=true,
	SNR::Real=2,
    model::Symbol = :linear,
    hidden_nodes::Int=10) 

    @assert k >= 0.0
    @assert SNR >= 0.0

	if k_missing_in_signal < 0 
		k_missing_in_signal = floor(Int, k*mean(canbemissing))
	end
	feature_names = Symbol.(names(data));
    nevermissing_features = feature_names[.!canbemissing]; missing_features = feature_names[canbemissing]
	setdiff!(feature_names, [:Test, :Id]); setdiff!(nevermissing_features, [:Test, :Id]); setdiff!(missing_features, [:Test, :Id]);

	k = min(k, length(feature_names))
    k_missing_in_signal = min(k_missing_in_signal, length(missing_features))
	k_non_missing = min(max(k - k_missing_in_signal, 0), length(nevermissing_features))
	# @show k, k_missing_in_signal, k_non_missing

    #Standardize
    newdata = standardize(data[:,feature_names])

    # Start creating input matrix
    X = zeros(nrow(newdata),0)

    #For nevermissing features, choose features
	if k_non_missing > 0
		support = shuffle(nevermissing_features)[1:k_non_missing]
		X = hcat(X, Matrix{Float64}(newdata[:,support]))
	end
    #For missing feautres, choose features
	if k_missing_in_signal > 0
	    support = shuffle(missing_features)[1:k_missing_in_signal]
	    X = hcat(X, Matrix{Float64}(newdata[:,support]))
	end
	if k_missing_in_signal > 0 && !mar
		support = shuffle(missing_features)[1:k_missing_in_signal]
		X = hcat(X, Matrix{Float64}(1.0 .* ismissing.(data_missing[:,support])))
	end
    #Add bias

	Y = zeros(nrow(newdata))
	if model == :linear 
		W = 2 .* rand(Base.size(X,2)) .- 1
		Y = X*W .+ randn(1)
	elseif model == :quad 
			W = 2 .* rand(Base.size(X,2)) .- 1
			Y = (X*W .+ randn(1) .- 1).^2
	elseif model == :break 
		W = 2 .* rand(Base.size(X,2)) .- 1
		Y = X*W .+ randn(1)
		Y .+= 3 .* (Y .> 0)
	elseif model == :linear 
		W = 2 .* rand(Base.size(X,2)) .- 1
		Y = X*W .+ randn(1)
	elseif model == :friedman || model == :tree
		if Base.size(X,2) < 5
			error("Friedman's model require at least 5 features")
		end 
		if  Base.size(X,2) > 5
			println("More than 5 features provided. FYI Friedman's model will only use 5 of them")
		end
		X = X[:,shuffle(1:Base.size(X,2))]
    	Y = 10 .* sin.(π .* X[:,1] .* X[:,2]) .+ 20 .* (X[:,3] .- 0.5).^2 .+ X[:,4] .+ 5 .* X[:,5]
	elseif model == :nn 
		W = 2*rand(hidden_nodes, Base.size(X,2))
		nn = Flux.Chain( 	Flux.Dense(W, # linear weights
                               randn(hidden_nodes), # bias of each hidden node
                               Flux.relu), # activation function
                    		Flux.Dense(2 * rand(1, hidden_nodes) .- 1, # linear weights
                               	randn(1), 	# bias of each hidden node
                               	x->x)) 		# activation function of output layer (identity)
    	Y = vec(nn(X'))
	end

    #Add noise
    noise = randn(nrow(newdata)); noise .*= norm(Y) / norm(noise) / SNR

    return Y .+ noise, k_non_missing+k_missing_in_signal, k_missing_in_signal
end

