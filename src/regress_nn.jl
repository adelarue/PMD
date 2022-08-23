###################################
### regress_nonlinear.jl
### Functions to perform neural network regression
### Authors: Arthur Delarue, Jean Pauphilet, 2022
###################################
using Flux 

"""
	Fit a neural network to the training data
"""
function regress_nn(Y::Array{Float64}, df::DataFrame;
						   hidden_nodes::Int=10)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	trainingset = try findall(df[:, :Test] .== 0) catch ; collect(1:nrow(df)) end
	# X = convert(Matrix, df[df[!, :Test] .== 0, cols])
	X = Matrix(df[trainingset, cols])
	y = convert(Array, Y[trainingset])
	
	data = Flux.Data.DataLoader((X', y'), batchsize=min(50,length(trainingset)), shuffle=true);
	# Defining our model, optimization algorithm and loss function
	m   = Flux.Chain(Flux.Dense(Base.size(X, 2), hidden_nodes, Flux.relu),
	                 Flux.Dense(hidden_nodes, 1, x -> x))
	opt = Flux.Momentum()
	loss(m, x, y) = sum(Flux.Losses.mse(m(x), y))
	evalcb() = @show(loss(m, X', y'))
	throttled_cb = Flux.throttle(evalcb, 5)
	# Training Method 1
	ps = Flux.params(m)
	epochs = 1000
	for i in 1:epochs
	    Flux.train!(loss, ps, data, opt, cb=throttled_cb)
	end
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