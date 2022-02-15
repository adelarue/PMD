###################################
### regress_nonlinear.jl
### Functions to perform neural network regression
### Authors: Arthur Delarue, Jean Pauphilet, 2022
###################################

"""
	Fit a neural network to the training data
"""
function regress_nonlinear(Y::Array{Float64}, df::DataFrame;
						   hidden_nodes::Int=10)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test])
	X = convert(Matrix, df[df[!, :Test] .== 0, cols])
	y = convert(Array, Y[df[!, :Test] .== 0])
	data = Flux.Data.DataLoader((X', y'), batchsize=50,shuffle=true);
	# Defining our model, optimization algorithm and loss function
	m   = Flux.Chain(Flux.Dense(Base.size(X, 2), hidden_nodes, Flux.relu),
	                 Flux.Dense(hidden_nodes, 1, x -> x))
	opt = Flux.Momentum()
	loss(x, y) = sum(Flux.Losses.mse(m(x), y))
	evalcb() = @show(loss(X', y'))
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
	X = convert(Matrix, df[:, cols])'
	return model(X)'
end

"""
	Evaluate fit quality on dataset
"""
function evaluate(Y::Vector, df::DataFrame, model::Chain)
	prediction = predict(df, model)
	trainmean = Statistics.mean(Y[df[:,:Test] .== 0])
	SST = sum((Y[df[:,:Test] .== 0] .- trainmean) .^ 2)
	OSSST = sum((Y[df[:,:Test] .== 1] .- trainmean) .^ 2)
	R2 = 1 - sum((Y[df[:,:Test] .== 0] .- prediction[df[:,:Test] .== 0]) .^ 2)/SST
	OSR2 = 1 - sum((Y[df[:,:Test] .== 1] .- prediction[df[:,:Test] .== 1]) .^ 2)/OSSST
	return R2, OSR2
end


