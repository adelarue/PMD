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

PHD.create_uci_datasets()
PHD.create_r_datasets()
