###################################
### create_sleep_datasets.jl
### Script to create regression datasets with missing values
### Uses the sleep dataset from the R VIM package,
### 	see https://cran.r-project.org/web/packages/VIM/VIM.pdf
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

using Pkg
Pkg.activate(".")

using Revise
using PHD

#PHD.create_uci_datasets()
PHD.create_r_datasets()

# using RCall, DataFrames, CSV, UCIData
# using Random, Printf

# function nan_to_missing!(df::DataFrame)
#     allowmissing!(df)

#     for i = 1:nrow(df), name in names(df)
#         if isnan(df[i, name])
#             df[i, name] = missing
#         end
#     end
# end

# function onehotencode(df)
#     categorical_cols = [k for k in names(df) if startswith(String(k),"C") || length(unique(df[:,k])) <= 5] #Categorical = starts with C or less than 5 unique values

#     if length(categorical_cols) > 0
#         long_cat = DataFrame(id=[], variable=[], value=[])
#         for c in categorical_cols
#             for i in 1:size(df,1)
#                 if !ismissing(df[i,c])
#                     push!(long_cat, [df[i,:id], string("C",String(c),"_",df[i,c]), 1])
#                 else
#                     push!(long_cat, [df[i,:id], string("C",String(c),"_","Missing"), 1])
#                 end
#             end
#         end

#         wide_cat = unstack(long_cat, :id, :variable, :value)
#         wide_cat = coalesce.(wide_cat,0)

#         select!(df, Not(categorical_cols))
#         return join(df, wide_cat, on=:id)
#     else
#         return df
#     end
# end

# R"library(missForest)"

# function impute_data(df, df_name)
#     path = "datasets/"*df_name*"/"

#     if !isdir(path)
#         mkdir(path)
#     end

#     Y = DataFrame(target=zeros(size(df,1)))
#     targetnames = [n for n in names(df) if occursin("target", string(n))]
#     istarget = length(targetnames) > 0
#     if istarget
#         for n in targetnames
#             Y[!,n] .= df[:, n]
#         end
#     end
#     select!(df, Not(intersect(names(df), targetnames)))

#     R"impute = missForest($df)"
#     R"imputeddf <- impute$ximp"

#     @rget imputeddf

#     idlist = [string("#",i) for i in 1:size(df,1)]
#     df[!,:Id] = idlist
#     imputeddf[!,:Id] = idlist

#     # Save the dataset
#     CSV.write(path*"X_missing.csv", df)
#     CSV.write(path*"X_full.csv", imputeddf)

#     if istarget
#         Y[!,:Id] = idlist
#         CSV.write(path*"Y.csv", Y)
#     end
# end

# Random.seed!(1515)

# # NUM_IMPUTATIONS = 20
# # TEST_FRACTION = 0.3

# folderlist = readdir("./results/")

# # Get dataset, and impute it using missForest
# R"library(VIM)"
# R"data(sleep, package='VIM')"
# R"sleep = as.data.frame(scale(sleep))"
# @rget sleep
# nan_to_missing!(sleep)
# impute_data(sleep, "sleep")

# for task in ["classification", "regression"]
# 	for n in UCIData.list_datasets(task)#[1:5]
#         if n ∉ folderlist || "Y.csv" ∉ readdir("./results/"*n*"/")
#             @show n
#     		df = UCIData.dataset(n)
#     		# if any([startswith(k,"C") for k in String.(names(df))])
#     			df = onehotencode(df) #One-hot encode categorical columns
#     		# end
#     		if any(.!completecases(df)) || any([endswith(k,"_Missing") for k in String.(names(df))])
#     			# @show n
#                 select!(df, Not(intersect(names(df), [:id])))

#     			if size(df, 2) > 1
#     				impute_data(df, n)
#     			end
#     		end
#         end
# 	end
# end
