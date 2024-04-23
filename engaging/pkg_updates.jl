@show ARGS
@show Symbol(ARGS[1])
# # println(Sys.CPU_NAME)

using Pkg
ENV["R_HOME"] = "/home/software/R/4.2.2/lib64/R/"
# # # @show get(ENV, "R_HOME", "")
Pkg.build("RCall")

Pkg.activate("..")
Pkg.update()
# # # ENV["R_HOME"] = "/home/software/R/4.2.2/lib64/R/"
# # # Pkg.build("RCall")

# ENV["CONDA_JL_VERSION"]="3.9"
# ENV["R_HOME"] = "/home/software/R/4.2.2/lib64/R/"
# ENV["PYTHON"] = "/home/software/python/3.9.4/bin/python"
# ENV["PYTHON"] = ""

# Pkg.add("PyCall")
# Pkg.build("PyCall")
# Pkg.add("ScikitLearn")

# math = PyCall.pyimport("math")
# math.pi
# PyCall.pyimport("sklearn")

# @sk_import ensemble: RandomForestRegressor

# # Pkg.update()
# # Pkg.precompile()
# # # Pkg.update()
# # # using PHD
@show readdir("/home/software/")
@show readdir("/home/software/R/4.2.2/lib64/R/lib/")
@show isfile("/home/software/R/4.2.2/lib64/R/lib/libR.so")

# # corefiles = filter(t -> startswith(t, "core."), readdir("/home/jpauph/Research/PHD/engaging/"))
# # for f in corefiles
# #     rm("/home/jpauph/Research/PHD/engaging/"*f, force=true)
# # end
