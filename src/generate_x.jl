###################################
### generate_x.jl
### Code to generate artificial independent variable
### Authors: XXXX

###################################
using Distributions, LinearAlgebra
function generate_x(n, d; rank::Int=4)
    B = randn(d,rank)
    Σ = B*B'
    Σ += Σ' ; Σ /= 2
    Σ += Matrix(max(eps(),eps()-2*eigmin(Σ))*I,d,d)
    while !isposdef(Σ)
        Σ += Matrix(eps()*I,d,d)
    end
    μ = randn(d)

    return DataFrame(rand(MvNormal(μ, Σ), n)', :auto)
end

function generate_missing(X::DataFrame; method::Symbol = :mar, p::Float64=0.1, kmissing::Int=Base.size(X,2))
    Xmissing = deepcopy(X)
    allowmissing!(Xmissing)
    missing_features = shuffle(1:Base.size(X,2))[1:kmissing]

    M = ones(Base.size(X,1),kmissing)
    if method == :mar 
        for i in 1:Base.size(X,1)
            for j in missing_features
                if rand() < p 
                    Xmissing[i,j] = missing 
                end 
            end 
        end
    elseif method == :censoring 
        for j in missing_features
            revind = rand() < .5 #Decide whether to cut low or high values
            cutoff = sort(X[:,j], rev=revind)[ceil(Int, (1-p)*Base.size(X,1))]
            for i in 1:Base.size(X,1)
                if (revind && X[i,j] .< cutoff) || (!revind && X[i,j] .> cutoff)
                    Xmissing[i,j] = missing 
                end 
            end 
        end
    end

    return Xmissing
end

