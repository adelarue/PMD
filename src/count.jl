###################################
### count.jl
### Functions to count number of missing values
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Count number of columns with at least one missing value (numeric)
"""
function count_missing_columns(df::DataFrame)
	counter = 0
	for name in names(df)
		if any(ismissing.(df[!, name]))
			counter += 1
		end
	end
	return counter
end

"""
	Given the name of a dataset in the datasets folder, count the number of missing
		columns with at least one missing value
"""
function count_missing_columns(datasetname::AbstractString)
	df = CSV.read("$(@__DIR__)/../datasets/$datasetname/X_missing.csv")
	return count_missing_columns(df)
end

"""
	Compute total percentage of missing values among columns with missing values
"""
function missing_percentage(df::DataFrame)
	counter = 0
	for i = 1:nrow(df), name in names(df)
		if ismissing(df[i, name])
			counter += 1
		end
	end
	return counter / (nrow(df) * count_missing_columns(df)) * 100.0
end

"""
	Given the name of a dataset in the datasets folder, count the number of missing
		columns with at least one missing value
"""
function missing_percentage(datasetname::AbstractString)
	df = CSV.read("$(@__DIR__)/../datasets/$datasetname/X_missing.csv")
	return missing_percentage(df)
end

"""
	Given a dataset name, return the dataset size
"""
function size(datasetname::AbstractString)
	df = CSV.read("$(@__DIR__)/../datasets/$datasetname/X_missing.csv")
	return Base.size(df)
end

"""
	List datasets meeting certain conditions
		- at least p_min missing columns
"""
function list_datasets(; p_min::Int = 0)
	dlist = String[]
	for dname in readdir("$(@__DIR__)/../datasets")
		if !startswith(dname, ".")
			if count_missing_columns(dname) >= p_min
				push!(dlist, dname)
			end
		end
	end
	return sort(dlist)
end

"""
	Return unique missingness patterns in the provided dataframe
"""
function missing_patterns(df::DataFrame)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test, :Y])
	return unique([ismissing.(convert(Vector, df[i, cols])) for i = 1:nrow(df)])
end
count_missing_patterns(df::DataFrame) = length(missing_patterns(df))

"""
	Return unique missingness patterns ordered by count
"""
function missing_patterns_countmap(df::DataFrame)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test, :Y])
	patterns = [ismissing.(convert(Vector, [df[i, j] for j in cols])) for i = 1:nrow(df)]

	patternmap = StatsBase.countmap(patterns)
	patterns = collect(keys(patternmap))
	counts = Int[]
	for pattern in patterns
		push!(counts, patternmap[pattern])
	end
	p = reverse(sortperm(counts))
	return patterns[p], counts[p]
end

"""
	Find rows with a unique missingness pattern
"""
function unique_missing_patterns(df::DataFrame)
	patterns, counts = missing_patterns_countmap(df)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test, :Y])
	unique_rows = Int[]
	for i = 1:nrow(df)
		pattern = ismissing.(convert(Vector, [df[i, j] for j in cols]))
		idx = findfirst(x -> x == pattern, patterns)
		if counts[idx] == 1
			push!(unique_rows, i)
		end
	end
	return unique_rows
end

"""
	Given a dataset with missing values, check if any columns with missing values have
		a counterpart column which perfectly matches the missingness indicator
	Returns a dictionary (column name with missing values) => [intrinsic indicator columns]
"""
function intrinsic_indicators(df::DataFrame; correlation_threshold::Real = 0.999)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test, :Y])
	d = Dict()
	for col in cols
		if any(ismissing.(df[!, col]))
			d[col] = []
			for col2 in setdiff(cols, [col])
				if !any(ismissing.(df[!, col2]))
					matrix = 0
					matrix = [Int.(ismissing.(df[!, col])) float.(convert(Vector, df[!, col2]))]
					correlation = cor(matrix)
					if abs(correlation[1,2]) >= correlation_threshold
						push!(d[col], col2)
					end
				end
			end
		end
	end
	return d
end

"""
	Split dataset into in-sample and out-of-sample using stratified sampling
	Returns a vector with value true if in test set and false otherwise
"""
function split_dataset(df::DataFrame; test_fraction::Real = 0.3, random::Bool = false)
	if !random
		return split_dataset_nonrandom(df, test_fraction=test_fraction)
	end
	cols = setdiff(Symbol.(names(df)), [:Id, :Test, :Y])
	patterns, counts = missing_patterns_countmap(df)
	patternidx = [findfirst(x -> x == ismissing.(convert(Vector, [df[i, j] for j in cols])),
	                        patterns) for i = 1:nrow(df)]
	train, test = MLDataPattern.stratifiedobs((eachindex(patternidx), patternidx), 1 - test_fraction)
	test_ind = falses(nrow(df))
	test_ind[test[1]] .= true
	return test_ind
end

"""
	Split dataset in non-random way, with as many missing values as possible
		in the testing set
"""
function split_dataset_nonrandom(df::DataFrame; test_fraction::Real = 0.3)
	cols = setdiff(Symbol.(names(df)), [:Id, :Test, :Y])
	patterns, counts = missing_patterns_countmap(df)
	patternidx = [findfirst(x -> x == ismissing.(convert(Vector, [df[i,j] for j in cols])),
	                        patterns) for i = 1:nrow(df)]
	p = reverse(sortperm(patterns, by=sum))
	test_ind = falses(nrow(df))
	test = Int[]
	for j in p
		datapoints = findall(x -> x == j, patternidx)
		if length(datapoints) + length(test) < test_fraction * length(test_ind)
			append!(test, datapoints)
		else
			num_datapoints = Int(ceil(test_fraction * length(test_ind)) - length(test))
			append!(test, shuffle(datapoints)[1:num_datapoints])
			break
		end
	end
	test_ind[test] .= true
	return test_ind
end

"""
	Swap missingness patterns in order to maximize total sum of missing values
"""
function optimize_missingness(X_missing::DataFrame, X_full::DataFrame)
    cols = setdiff(Symbol.(names(X_missing)), [:Id])
    patterns, counts = missing_patterns_countmap(X_missing)
    model = Model(with_optimizer(Gurobi.Optimizer, TimeLimit=10, OutputFlag=0))
    @variable(model, z[i = 1:nrow(X_full), j=eachindex(patterns)], Bin)
    @constraint(model, [i = 1:nrow(X_full)], sum(z[i, j] for j = eachindex(patterns)) == 1)
    @constraint(model, [j = eachindex(patterns)], sum(z[i, j] for i = 1:nrow(X_full)) == counts[j])
    @objective(model, Max, sum(z[i, j] * sum(X_full[i, name] * patterns[j][k] for (k, name) in enumerate(cols))
                               for i = 1:nrow(X_full), j = eachindex(patterns)))
    optimize!(model)
    new_X_missing = deepcopy(X_full)
    allowmissing!(new_X_missing)
    for i = 1:nrow(X_full), j=eachindex(patterns)
        if value(z[i, j]) > 0.5
            for (k, name) in enumerate(cols)
                if patterns[j][k]
                    new_X_missing[i, name] = missing
                end
            end
        end
    end
    return new_X_missing
end
