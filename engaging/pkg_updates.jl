# println(Sys.CPU_NAME)

using Pkg
ENV["R_HOME"] = "/home/software/R/4.2.2/lib64/R/"
# # @show get(ENV, "R_HOME", "")
# # Pkg.build("RCall")

Pkg.activate("..")
# # ENV["R_HOME"] = "/home/software/R/4.2.2/lib64/R/"
# # Pkg.build("RCall")

# Pkg.update()
Pkg.precompile()
# # Pkg.update()
# # using PHD
# # @show readdir("/home/software/")
# # @show readdir("/home/software/R/4.2.2/lib64/R/lib/")
# # @show isfile("/home/software/R/4.2.2/lib64/R/lib/libR.so")

# corefiles = filter(t -> startswith(t, "core."), readdir("/home/jpauph/Research/PHD/engaging/"))
# for f in corefiles
#     rm("/home/jpauph/Research/PHD/engaging/"*f, force=true)
# end
