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
function standardize(data::DataFrame; separate_test::Bool = true)
	@assert count_missing_columns(data) == 0
	truenames = setdiff(names(data), [:Test])
	data_for_stats = data
	if separate_test
		data_for_stats = filter(row -> row[:Test] == 0, data)
	end
	μ = [mean(data_for_stats[!, name]) for name in truenames]
	σ = [std(data_for_stats[!, name]) for name in truenames]
	# if a stddev is 0, the column is constant, and we should simply not rescale it
	σ[findall(σ .== 0)] .= 1
	newdata = DataFrame()
	for (i, name) in enumerate(truenames)
		newdata[!, name] = (data[!, name] .- μ[i]) ./ σ[i]
	end
	newdata[!, :Test] = data[!, :Test]
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
function linear_y(data::DataFrame; soft_threshold::Real=0.1, SNR::Real=4)
	@assert soft_threshold >= 0.0
	n, p = Base.size(data)
	# sample coefficients
	wtrue = softthresholding.(randn(p), λ=soft_threshold)
	btrue = rand(1)
	# remove coefficient for Test column
	test_index = findfirst(names(data) .== :Test)
	wtrue[test_index] = 0.
	# standardize and generate Y
	newdata = standardize(data, separate_test = false)
	Y = Matrix{Float64}(newdata) * wtrue .+ btrue
	# add noise
	noise = randn(nrow(newdata))
	noise .*= norm(Y) / norm(noise) / SNR
	return Y .+ noise
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
