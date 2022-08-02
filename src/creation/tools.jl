###################################
### creation/tools.jl
### Helper functions for dataset creation
### Authors: Arthur Delarue, Jean Pauphilet, 2019
###################################

"""
	Convert NaN to missing
"""
function nan_to_missing!(df::DataFrame)
    allowmissing!(df)

    for i = 1:nrow(df), name in names(df)
        if isnan(df[i, name])
            df[i, name] = missing
        end
    end
end

"""
    Identify columns wrongly encoded as strings 
"""
function string_to_float_fix!(df::DataFrame)
    keep_obs = trues(Base.size(df,1)) #Returns list of problematic observations where re-encoding failed
    for j in names(df)
        if Symbol(j) != :Id && (eltype(df[:,j]) == String || eltype(df[:,j]) == Union{Missing,String})
            newcol = tryparse.(Float64, df[:,j])
            keep_obs[newcol .== nothing] .= false ##If re-encoding failed -> remove observation
            newcol = convert(Array{Union{Float64,Missing,Nothing}}, newcol) #Allow for missingness in newcol
            newcol[newcol .== nothing] .= missing #Re-encode nothing (failed parsing) into missing
            newcol = convert(Array{Union{Float64,Missing}}, newcol)
            df[!,j] .= newcol #Update df
        end
    end

    return keep_obs
end

"""
	One-hot encode categorical variables
	Using UCIData convention, categorical variables start with the letter 'C',
		or contain <=5 unique values
"""
function onehotencode(df)
    categorical_cols = [k for k in names(df) if startswith(String(k),"C") || length(unique(df[:,k])) <= 5] #Categorical = starts with C or less than 5 unique values

    if length(categorical_cols) > 0
        long_cat = DataFrame(id=[], variable=[], value=[])
        for c in categorical_cols
            for i in 1:nrow(df)
                if !ismissing(df[i,c])
                    push!(long_cat, [df[i,:id], string("C",String(c),"_",df[i,c]), 1])
                else
                    push!(long_cat, [df[i,:id], string("C",String(c),"_","Missing"), 1])
                end
            end
        end

        wide_cat = unstack(long_cat, :id, :variable, :value)
        wide_cat = coalesce.(wide_cat,0)

        select!(df, Not(categorical_cols))
        return join(df, wide_cat, on=:id)
    else
        return df
    end
end

"""
	Given a dataset (with one-hot-encoded categorical variables) and name:
		1) Put target variables in separate dataframe
		2) Impute missing values using missForest on the non-target variables (ground truth)
		3) Save X_missing, X_full, and Y to csv
"""
function impute_data(df, df_name)
    path = "datasets/"*df_name*"/"

    if !isdir(path)
        mkdir(path)
    end

    Y = DataFrame(target=zeros(nrow(df)))
    targetnames = [n for n in names(df) if occursin("target", string(n))]
    istarget = length(targetnames) > 0
    if istarget
        for n in targetnames
            Y[!,n] .= df[:, n]
        end
    end
    select!(df, Not(intersect(names(df), targetnames)))

    load_R_library("missForest")
    R"impute = missForest($df)"
    R"imputeddf <- impute$ximp"

    @rget imputeddf

    idlist = [string("#",i) for i in 1:nrow(df)]
    df[!,:Id] = idlist
    imputeddf[!,:Id] = idlist

    # Save the dataset
    CSV.write(path*"X_missing.csv", df)
    CSV.write(path*"X_full.csv", imputeddf)

    if istarget
        Y[!,:Id] = idlist
        CSV.write(path*"Y.csv", Y)
    end
end
