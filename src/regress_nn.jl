###################################
### regress_nonlinear.jl
### Functions to perform neural network regression
### Authors: XXXX

###################################
using Flux 

"""
	Fit a neural network to the training data
"""
function regress_nn(Y::Array{Float64}, df::DataFrame;
						   hidden_nodes::Int=10, maxepochs::Int=1000, batchsize::Int=10, timeLimit::Int=300)
	
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	# X = convert(Matrix, df[df[!, :Test] .== 0, cols])
	X = Matrix{Float64}(df[trainingset, cols])
	y = convert(Array{Float64}, Y[trainingset])
	
	data = Flux.Data.DataLoader((X', y'), batchsize=min(batchsize,length(trainingset)), shuffle=true);
	# Defining our model, optimization algorithm and loss function
	m   = Flux.Chain(Flux.Dense(Base.size(X, 2), hidden_nodes, Flux.relu, bias=true),
	                 Flux.Dense(hidden_nodes, 1, x -> x, bias=true))
	opt = Flux.AdaGrad()
	loss(x, y) = Flux.Losses.mse(m(x), y)
	# evalcb() = @show(loss(X', y'))
	# throttled_cb = Flux.throttle(evalcb, 5)
	ps = Flux.params(m)
	
	mse_old = mean((y .- mean(y)).^2)
	n_noimprov = 0
	start_time = time()
	for epoch in 1:maxepochs
		Flux.train!(loss, ps, data, opt)
		if epoch % 10 == 0
			mse_new = mean((y .- m(X')).^2)

			if mse_new > mse_old
				n_noimprov += 1
			else
				n_noimprov = 0
			end

			stop_if = (n_noimprov > 2) || (abs(mse_old - mse_new) < 1e-3*mse_old) #No improvement for 30 iterations (3 batches of 10) OR less than 0.1% improvement
			
			mse_old = mse_new
			if stop_if || time() - start_time > timeLimit
				break
			end
		end
	end
	# Flux.@epochs maxepochs Flux.train!(loss, ps, data, opt, cb=throttled_cb)
	
	return m
end


function regress_nn(Y::BitArray, df::DataFrame;
	hidden_nodes::Int=10, maxepochs::Int=1000, batchsize::Int=10, timeLimit::Int=300)

	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	# X = convert(Matrix, df[df[!, :Test] .== 0, cols])
	X = Matrix{Float64}(df[trainingset, cols])
	y = convert(Array{Float64}, Y[trainingset])

	data = Flux.Data.DataLoader((X', y'), batchsize=min(batchsize,length(trainingset)), shuffle=true);
	
	# Defining our model, optimization algorithm and loss function
	m   = Flux.Chain(	Flux.Dense(Base.size(X, 2), hidden_nodes, Flux.relu, bias=true),
						Flux.Dense(hidden_nodes, 1, NNlib.sigmoid, bias=true))
	opt = Flux.AdaGrad()
	loss(x, y) = Flux.Losses.binarycrossentropy(m(x), y)
	# evalcb() = @show(loss(X', y'))
	# throttled_cb = Flux.throttle(evalcb, 5)
	ps = Flux.params(m)

	mse_old = Flux.Losses.binarycrossentropy(mean(y)*ones(Base.size(X,1)), y)
	
	n_noimprov = 0
	start_time = time()
	for epoch in 1:maxepochs
		Flux.train!(loss, ps, data, opt)
		if epoch % 10 == 0
			mse_new = Flux.Losses.binarycrossentropy(vec(m(X')), y)

			if mse_new > mse_old
				n_noimprov += 1
			else
				n_noimprov = 0
			end

			stop_if = (n_noimprov > 2) || (abs(mse_old - mse_new) < 1e-3*mse_old) #No improvement for 30 iterations (3 batches of 10) OR less than 0.1% improvement
			mse_old = mse_new
			if stop_if || time() - start_time > timeLimit
				break
			end
		end
	end
	# Flux.@epochs maxepochs Flux.train!(loss, ps, data, opt, cb=throttled_cb)
	 println()
	return m
end

"""
	Get neural net predictions on dataset
"""
function predict(df::DataFrame, model::Chain)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	# X = convert(Matrix, df[:, cols])'
	X = Matrix(df[:, cols])'
	return model(X)'
end