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

	μ = [mean(data[.!ismissing.(data[!, name]), name]) for name in truenames]
	σ = [std(data[.!ismissing.(data[!, name]), name]) for name in truenames]
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

"""
	Generate Y, linearly dependent on the features of X. More precisely, the true model has
		(possibly soft-thresholded) standard normal coefficients and a U([0, 1]) bias, to
		which we add zero-mean Gaussian noise
	Args:
		- data: DataFrame with fully observed values
	Keyword args:
		- soft_threshold: soft threshold subtracted in absolute value from all true coefficients
			(see softthresholding function)
		- SNR: signal-to-noise ratio
	Returns:
		- a vector of length nrow(data) with the values of Y
"""
function linear_y(data::DataFrame, data_missing::DataFrame;
	k::Real=10, SNR::Real=4,
    canbemissing=falses(Base.size(data,2)), #indicates which features can be missing
	mar::Bool=true,
    k_missing_in_signal::Int=0) #indicates the number of potentially missing features in signal

    @assert k >= 0.0
    @assert SNR >= 0.0

	feature_names = Symbol.(names(data));
    nevermissing_features = feature_names[.!canbemissing]; missing_features = feature_names[canbemissing]
	setdiff!(feature_names, [:Test, :Id]); setdiff!(nevermissing_features, [:Test, :Id]); setdiff!(missing_features, [:Test, :Id]);

	k = min(k, length(feature_names))
    k_missing_in_signal = min(k_missing_in_signal, length(missing_features))
	k_non_missing = min(max(k - k_missing_in_signal, 0), length(nevermissing_features))

    #Standardize
    newdata = standardize(data[:,feature_names])

    Y = zeros(nrow(newdata))
    #For nevermissing features, choose then generate
	@show k_non_missing
	if k_non_missing > 0
		support = shuffle(nevermissing_features)[1:k_non_missing]
		w1 = 2*rand(k_non_missing) .- 1
		Y += Matrix{Float64}(newdata[:,support])*w1
	end
    #For missing feautres, choose
	@show k_missing_in_signal
	if k_missing_in_signal > 0
	    support = shuffle(missing_features)[1:k_missing_in_signal]
	    w2 = 2*rand(k_missing_in_signal) .- 1
	    Y += Matrix{Float64}(newdata[:,support])*w2
		# if !mar
		# 	w2m = 2*rand(k_missing_in_signal) .- 1
		# 	Y += Matrix{Float64}(1.0 .* ismissing.(data_missing[:,support]) )*w2m
		# end
	end
	if k_missing_in_signal > 0 && !mar
		support = shuffle(missing_features)[1:k_missing_in_signal]
		w2m = 2*rand(k_missing_in_signal) .- 1
		Y += Matrix{Float64}(1.0 .* ismissing.(data_missing[:,support]) )*w2m
	end
    #Add bias
    btrue = randn(1); Y .+= btrue
    #Add noise
    noise = randn(nrow(newdata)); noise .*= norm(Y) / norm(noise) / SNR

    return Y .+ noise, k_non_missing+k_missing_in_signal, k_missing_in_signal
end

"Sigmoid function"
sigmoid(x::Real) = 1 / (1 + exp(-1 * x))

"Generate binary Y"
function binary_y(data::DataFrame, data_missing::DataFrame;
				  k::Real=10, SNR::Real=4,
    			  canbemissing=falses(Base.size(data,2)), #indicates which features can be missing
				  mar::Bool=true,
    			  k_missing_in_signal::Int=0,
    			  sigmoid_threshold::Real=0.5)
	Y, k1, k2 = linear_y(data, data_missing, k=k, SNR=SNR, canbemissing=canbemissing,
	                     mar=mar, k_missing_in_signal=k_missing_in_signal)
	@show length(Y)
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
