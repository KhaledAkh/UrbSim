# This file contains the code for the analysis used for the assignment of Urban Simulation module
# It is intended to be run line by line (or block by block) in the console similar to a Jupyter notebook ot R studio

from math import sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr


# set up the metric calculations
def CalcRSqaured(observed, estimated):
    r, p = pearsonr(observed, estimated)
    R2 = r ** 2
    return R2


def CalcRMSE(observed, estimated):
    res = (observed - estimated) ** 2
    RMSE = round(sqrt(res.mean()), 3)
    return RMSE


# Importing the Origin Destination Data
cdatasub = pd.read_csv("london_flows.csv")

# Chop out the intra-borough flows and zeros in the important data
cdatasub = cdatasub[cdatasub["station_origin"] != cdatasub["station_destination"]]
cdatasub = cdatasub.drop(cdatasub[cdatasub['jobs'] == 0].index)
cdatasub = cdatasub.drop(cdatasub[cdatasub['population'] == 0].index)
cdatasub = cdatasub.drop(cdatasub[cdatasub['flows'] == 0].index)
print(sum(cdatasub["flows"]))

# Random uncalibrated unconstrained model
# run the model with parameters 1 and store of the new flow estimates in a new column
T1 = cdatasub["population"] ** 1 * cdatasub["jobs"] ** 1 * cdatasub["distance"] ** -2
k = sum(cdatasub["flows"]) / sum(T1)
cdatasub["unconstrainedEst1"] = round(k * T1, 0).astype(int)

# check that the sum of these estimates make sense
print(sum(cdatasub["unconstrainedEst1"]))
# check the error
CalcRSqaured(cdatasub["flows"], cdatasub["unconstrainedEst1"])
CalcRMSE(cdatasub["flows"], cdatasub["unconstrainedEst1"])

## Production Constrained model
# creating log values of attraction and distance parameters
cdatasub['log_jobs'] = np.log(cdatasub['jobs'])
cdatasub['log_dist'] = np.log(cdatasub['distance'])

# create the formula (the "-1" indicates no intercept in the regression model).
formula = 'flows ~ station_origin + log_jobs + log_dist-1'
# run a production constrained sim
prodSim = smf.glm(formula=formula, data=cdatasub, family=sm.families.Poisson()).fit()
# let's have a look at it's summary
print(prodSim.summary())

O_i = pd.DataFrame(cdatasub.groupby(["station_origin"])["flows"].agg(np.sum))
O_i.rename(columns={"flows": "O_i"}, inplace=True)
cdatasub = cdatasub.merge(O_i, on="station_origin", how="left")

D_j = pd.DataFrame(cdatasub.groupby(["station_destination"])["flows"].agg(np.sum))
D_j.rename(columns={"flows": "D_j"}, inplace=True)
cdatasub = cdatasub.merge(D_j, on="station_destination", how="left")

coefs = pd.DataFrame(prodSim.params)
coefs.dropna()
coefs.reset_index(inplace=True)
coefs.rename(columns={0: "alpha_i", "index": "coef"}, inplace=True)
to_repl = ["station_origin", "[", "]"]
for x in to_repl:
    coefs["coef"] = coefs["coef"].str.replace(x, "")

# then once you have done this you can join them back into the dataframes
cdatasub = cdatasub.merge(coefs, left_on="station_origin", right_on="coef", how="left")
cdatasub.drop(columns=["coef"], inplace=True)
# check this has worked
len(cdatasub['station_origin'].unique())

alpha_i = prodSim.params[0:-2]
gamma = prodSim.params[-2]
beta = -prodSim.params[-1]

cdatasub["prodsimest1"] = np.exp(cdatasub["alpha_i"] + gamma * cdatasub["log_jobs"]
                                 - beta * cdatasub["log_dist"])

# first round the estimates
cdatasub["prodsimest1"] = round(cdatasub["prodsimest1"], 0)
# now we can create a pivot tabel to turn the paired list into a matrix, and compute the margins as well
cdatasubmat3 = cdatasub.pivot_table(values="prodsimest1", index="station_origin", columns="station_destination",
                                    aggfunc=np.sum, margins=True)

cdatasubmat = cdatasub.pivot_table(values="flows", index="station_origin", columns="station_destination",
                                   aggfunc=np.sum, margins=True)
print(cdatasubmat)
print(cdatasubmat3)

CalcRSqaured(cdatasub["flows"], cdatasub["prodsimest1"])
CalcRMSE(cdatasub["flows"], cdatasub["prodsimest1"])

## Applying scnario one by reducing Canary Wharf jobs to half
print(cdatasub[cdatasub['station_destination'] == 'Canary Wharf'].iloc[0])


def new_jobs(row):
    if row["station_destination"] == "Canary Wharf":
        val = row["jobs"] / 2
    else:
        val = row["jobs"]
    return val


cdatasub["jobs_sA"] = cdatasub.apply(new_jobs, axis=1)

# calculate some new wj^alpha and d_ij^beta values
dist_beta = cdatasub["distance"] ** -beta
Dj3_gamma = cdatasub["jobs_sA"] ** gamma
# calcualte the first stage of the Ai values
cdatasub["Ai1"] = Dj3_gamma * dist_beta
# now do the sum over all js bit
A_i = pd.DataFrame(cdatasub.groupby(["station_origin"])["Ai1"].agg(np.sum))
# now divide into 1
A_i["Ai1"] = 1 / A_i["Ai1"]
A_i.rename(columns={"Ai1": "A_i2"}, inplace=True)
# and write the A_i values back into the dataframe
cdatasub = cdatasub.merge(A_i, left_on="station_origin", right_index=True, how="left")

# to check everything works, recreate the original estimates
cdatasub["prodsimest_scA"] = cdatasub["A_i2"] * cdatasub["O_i"] * Dj3_gamma * dist_beta
# round
cdatasub["prodsimest_scA"] = round(cdatasub["prodsimest_scA"], 0)

cdatasubmat_scA = cdatasub.pivot_table(values="prodsimest_scA", index="station_origin", columns="station_destination",
                                       aggfunc=np.sum, margins=True)
print(cdatasubmat_scA)

print(cdatasubmat_scA['Canary Wharf'])
print(cdatasubmat['Canary Wharf'])

# Scnario B1
cdatasub["prodsimestb1"] = np.exp(cdatasub["alpha_i"] + gamma * np.log(cdatasub["jobs_sA"])
                                  - (beta * 1.1) * cdatasub["log_dist"])
cdatasub["prodsimestb1"] = round(cdatasub["prodsimestb1"], 0)
# now we can convert the pivot table into a matrix
cdatasubmatb1 = cdatasub.pivot_table(values="prodsimestb1", index="station_origin", columns="station_destination",
                                     aggfunc=np.sum, margins=True)
print(cdatasubmatb1)

# Scnario B2
cdatasub["prodsimestb2"] = np.exp(cdatasub["alpha_i"] + gamma * np.log(cdatasub["jobs_sA"])
                                  - (beta * 1.2) * cdatasub["log_dist"])
cdatasub["prodsimestb2"] = round(cdatasub["prodsimestb2"], 0)
# now we can convert the pivot table into a matrix
cdatasubmatb2 = cdatasub.pivot_table(values="prodsimestb2", index="station_origin", columns="station_destination",
                                     aggfunc=np.sum, margins=True)
print(cdatasubmatb2)
