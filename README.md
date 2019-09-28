# PHD
Predictions with Hidden Data

## Quick start

Take a look at the example notebooks.

Note: to avoid pushing metadata to Github, try to save a "clean" notebook with all outputs cleared.

## Creating datasets

Run `julia create_sleep_datasets.jl` to create some data files in the `datasets/` folder. For now, parameters can only be modified directly in the source file.

Conventions: all datasets must have a column called `Test` which is 1 if the row belongs to the test set and 0 if it belongs to the training set. Additionally, all datasets must denote the dependent variable as `Y`.

## Imputation and regression

- `impute.jl` defines functions to impute missing values using MICE, and to impute missing values with zeros.
- `regress.jl` defines functions to perform linear regression and evaluate fit quality.
- `augment.jl` defines functions to add missingness-related features to the data.
