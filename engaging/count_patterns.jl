using Pkg
Pkg.activate("..")

using Revise
using PHD

using CSV, DataFrames


datasets = readdir("../datasets")
patterndata = DataFrame(Name = datasets, Num_Patterns = zeros(Int, length(datasets)),
	                    n = zeros(Int, length(datasets)), p = zeros(Int, length(datasets)),
	                    p_miss = zeros(Int, length(datasets)),
	                    Most_Frequent_Count = zeros(Int, length(datasets)),
	                    Least_Frequent_Count = zeros(Int, length(datasets)),
	                    Fully_Observed_Count = zeros(Int, length(datasets)))

for i = 1:nrow(patterndata)
	df = CSV.read("../datasets/" * patterndata[i, :Name] * "/X_missing.csv", missingstrings=["", "NaN"])
	patterns, counts = PHD.missing_patterns_countmap(df)
	patterndata[i, :n] = nrow(df)
	patterndata[i, :p] = length(setdiff(names(df), [:Id, :Y, :Test]))
	patterndata[i, :p_miss] = PHD.count_missing_columns(df)
	patterndata[i, :Num_Patterns] = length(patterns)
	patterndata[i, :Most_Frequent_Count] = counts[1]
	patterndata[i, :Least_Frequent_Count] = counts[end]
	fully_observed_idx = findfirst(x -> all(.!x), patterns)
	if fully_observed_idx != nothing
		patterndata[i, :Fully_Observed_Count] = counts[fully_observed_idx]
	else
		patterndata[i, :Fully_Observed_Count] = 0
	end
	@show patterndata[i, :]
end

CSV.write("../results/pattern_counts.csv", patterndata)
