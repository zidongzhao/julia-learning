# Import Turing and Distributions.
using Turing, Distributions
# GLM for comparison
using GLM
# Import RDatasets.
using RDatasets
# Import MCMCChains, Plots, and StatPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots
# Functionality for splitting and normalizing the data.
using MLDataUtils: shuffleobs, splitobs, rescale!
# Functionality for evaluating the model predictions.
using Distances
# set random seeds
using Random
Random.seed!(12212020)

# import RDatasets
df = RDatasets.dataset("datasets","ToothGrowth");
# Dummy code the Supp column
df.Supp_num = convert.(Float64, df.Supp .== "VC")

# split train-test
train, test = splitobs(shuffleobs(df), at=.7)

# put DV and IV into separate vars
function split_iv_dv(df)
    y = Array(df.Len)
    X = Array(df[:,["Supp_num","Dose"]])
    return y, X
end
y_train, X_train = split_iv_dv(train)
y_test, X_test = split_iv_dv(test)

# OLS for comparison
ols = lm(@formula(Len ~ Supp * Dose), train)

# define the generative function
@model function linear_regression(y, X)
    # construct interaction term
    X = [X X[:,1] .* X[:,2]]
    # prior for intercept
    b0 ~ Normal(0,3)
    # prior for betas, mvnormal with 0 means
    bs ~ MvNormal(size(X,2), 3)
    # error variance prior
    e ~ InverseGamma(2,3)

    ## condition on obs
    mu = b0 .+ X * bs
    y ~ MvNormal(mu, e)
end

# prior check, plot sample from prior
prior_sample = sample(
    linear_regression(y_train, X_train),
    Prior(),
    1000
)
plot(prior_sample)
describe(prior_sample)

# NUTS sampling posterior
ch = sample(
    linear_regression(y_train, X_train),
    NUTS(.65),
    1000
)

# visualize the samples
plot(ch)
# summarize the samples
describe(ch)

## check against hold-out data
