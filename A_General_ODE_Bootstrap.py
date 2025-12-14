####################################################################################
# This function is meant to be a (hopefully) accessible generic bootstrap function #
# that can be implemented for any system of ordinary differential equations. It    #
# should be noted that you, as the user, will need to provide several things:      #
#                                                                                  #
# REQUIREMENTS:    1. ODE model (You need to pre-specify your ODE model)           #
#                                                                                  #
#                  2. This function works under the assumption that time           #
#                     is a component of your ODE model (i.e., time series          #
#                     data), so this will only work for time series data.          #
#                                                                                  #
#                  3. If you are working with multiple datasets or groups          #
#                     (such as group A vs group B vs group C, etc.), then          #
#                     you need to pre-separate your full dataframe into            #
#                     the appropriate groups (so that each group is its own        #
#                     dataframe).                                                  #
#                                                                                  #
#                  4. Initial conditions for all equations in ODE model            #
####################################################################################

# ---------------------------------------------------------------------------------#

############################
# Required python packages #
############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from collections import Counter
from joblib import Parallel, delayed
import os

# ---------------------------------------------------------------------------------#

############################################################
# Pulling necessary time-series information for ODE solver #
############################################################

# df -> user's dataframe
# time_column -> name of the column containing the time-series information

def timeseries_pull(df, time_column):

    # This raises an error if the column name with times is not specified correctly
    if time_column not in df.columns:
        raise ValueError(f"The column name '{time_column}' isn't recognized... most likely because it does not match what you have in your dataframe. Double-check the name of the column containing the times :)")

    # These two lines are pulling out all instances of the time data (sampleTimes) and the unique timepoints (uniqueTimes)
    sampleTimes = df[time_column].values.astype(float)
    uniqueTimes = np.unique(sampleTimes)

    return sampleTimes, uniqueTimes

# ---------------------------------------------------------------------------------#

###########################################
# Pulling observation data for ODE solver #
###########################################

# df -> user's dataframe
# data_columns -> list (in the form ["Variable A", "Variable B", etc.]) of column name(s) containing observed data

def observedData_pull(df, data_columns):

    # This raises an arror if the column(s) with the observed data is not passed to the function as a list
    if not isinstance(data_columns, list):
        raise TypeError(f"{data_columns} needs to be passed as a list, even if there is only one observed data column. For example:\n "
                        "One variable -> ['Variable A'] \n" 
                        "More than one variable -> ['Variable A', 'Variable B', etc.]")

    # This raises an error if the column name(s) with the observed data is not specified correctly
    for col in data_columns:
        if col not in df.columns:
            raise ValueError(f"The column name '{col}' isn't recognized... most likely because it does not match what you have in your dataframe. Double-check the name of the column(s) containing the observed data :)")

    # This line puts all extracted columns into a dictionary that we can later iterate through (to keep things generic)
    obs_data = [df[col].values.astype(float) for col in data_columns]

    return obs_data

# ---------------------------------------------------------------------------------#

########################################
# Defining cost function for ODE model #
########################################

# params -> List of parameter names user wants to estimate
# params_trans -> List of transformations user may want to apply to parameters
#                 NOTE: Order of parameter transformations needs to match parameter
#                       order specified in 'params'
# ODE_model -> User specified ODE model
# init_cond -> List of initial conditions
# obs_data -> List of observed data names
# sample_times -> List of all sample time points
# unique_times -> List of all unique time points

def CostFunction(params, params_trans, ODE_model, init_cond, obs_data, sample_times, unique_times, ODE_names=None, return_all=False):

    # These lines undo log_10 transformations if specified by the user, otherwise, the original parameter remains unchanged (linear line)
    # This line creates an empty list to store un-transformed/original parameters
    transformed_params = []

    # This for loop works through parameters specified by user to undo log_10 transforms or leave parameters unchanged (as specified by the user).
    # Specification for transformations, or lack thereof, should be entered as a list: params_trans = ['linear', 'log10', 'log10', 'linear', etc.]
    # NOTE: Parameter transformation specifications must match the exact order parameters are specified in. So, if you have parameters a, b, c, and d,
    #       you enter the parameters as params = ['a', 'b', 'c', 'd'], and you log-transformed b and c, then you would need to specify the
    #       order for params_trans as: params_trans = ['linear', 'log10', 'log10', 'linear'] to un-log transform b and c for the ODE solver.
    for parameters, rule in zip(params, params_trans):
        if rule == "linear":
            transformed_params.append(parameters)
        elif rule == "log10":
            transformed_params.append(10**parameters)
        else:
            raise ValueError(f"In your transform list '{rule}', you entered a transform that is not supported, or you misspecified a transform that is supported. The only transforms supported at this time are 'linear' (which leaves your parameter unchanged) or 'log10' (which un-logs a parameter you might have had on the log10 scale). Also, since you are getting this error, double-check that the parameter order you specified matches the same transform order you need :)")

    # This line ensures only one time vector gets fed into the ODE solver
    if isinstance(unique_times, (list, tuple)):
        t_ode = np.unique(np.concatenate(unique_times))
    else:
        t_ode = np.asarray(unique_times).flatten()
    
    # This line solves your ODE system
    ODEsolution = odeint(ODE_model, init_cond, t_ode, args=tuple(transformed_params))

    # This line calculates the number of ODE solutions present
    n_ODE_sol = ODEsolution.shape[1]
    
    # This line works through the solved ODE equations and pulls the solutions for each equation
    # NOTE: The order of your solutions for your system of equations will match the same exact order of returned equations specifed in your model (ODE_model)
    ODE_outputs = [ODEsolution[:,i] for i in range(n_ODE_sol)]

    # This line creates a dictionary for the unique time points to the ODE solutions so we can map outputs to the sample time points
    dictTimes = []
    for i in range(n_ODE_sol):
        dictTimes.append(dict(zip(unique_times[i], ODEsolution[:, i])))
        
    #[dict(zip(unique_times, ODE_outputs[i])) for i in range(n_ODE_sol)] OLD-INCORRECT???

    # This line maps the model predictions to the correct sample times
    map_obs_data = []
    for i in range(n_ODE_sol):
        mapped = np.array([dictTimes[i][time] for time in sample_times[i]])
        map_obs_data.append(mapped)
    #[np.array([dictTimes[i][time] for time in sample_times[i]]) for i in range(n_ODE_sol)]

    # This line counts how many of each unique time point you have
    timeFreqs = []
    for i in range(n_ODE_sol):
        freq = Counter(sample_times[i])
        timeFreqs.append(freq)
    #[Counter(sample_times[i]) for i in range(n_ODE_sol)]

    # This line DOES SOMETHING
    mapTimeFreqs = [np.array([timeFreqs[i][time] for time in sample_times[i]]) for i in range(n_ODE_sol)]

    # This line adds a small epsilon so that log_10(0) isn't an issue
    eps = 1e-8

    # This line ensures mapped ODE outputs aren't log_10(0)
    map_obs_data = [np.maximum(map_obs_data[i], eps) for i in range(n_ODE_sol)]

    # This line calculates all individual RMSLE (root mean squared log error) for each ODE solution
    RMSLEs = []
    for i in range(n_ODE_sol):
        rmsle_i = np.sqrt(np.sum((np.log10(map_obs_data[i]) - np.log10(obs_data[i]))**2 / mapTimeFreqs[i]))
        RMSLEs.append(rmsle_i)

    Total_RMSLE = sum(RMSLEs)
    
    if return_all:

        # Optional argument to append ODE equation label to corresponding RMSLE
        if ODE_names is not None:
            if len(ODE_names) != n_ODE_sol:
                raise ValueError("The number of ODE equations/names entered in ODE_names needs to match the number of ODE outputs your ODE_model returns (so maybe double-check your list ODE_names) :)")
            
            else:
                RMSLEs_results = {f"{ODE_eq}": rmsle_i for ODE_eq, rmsle_i in zip(ODE_names, RMSLEs)}
        else:
            RMSLEs_results = RMSLEs
        
        return Total_RMSLE, RMSLEs_results
        
    return Total_RMSLE

# ---------------------------------------------------------------------------------#

####################################################
# Defining bootstrap function to decrease run-time #
####################################################

def boot_run(b, params_trans, ODE_model, init_cond, obs_data, sample_times, unique_times, bounds, maxiter, tol, rmsle_return, ODE_names):
    
    obs_b = []
    sampleTime_b = []
    uniqueTime_b = []

    # This for loop creates a bootstrapped sample to then run through DE
    for obs_i, sample_i, unique_i in zip(obs_data, sample_times, unique_times):

        # These lines reshape the data so that bootstrapped samples can be taken from the 
        obs_i = np.asarray(obs_i).reshape(-1)
        sample_i = np.asarray(sample_i).reshape(-1)      
        unique_i = np.asarray(unique_i).reshape(-1) 
                
        obs_boot_i = np.array([np.random.choice(obs_i[sample_i == t_i], size=1) for t_i in unique_i]).flatten()

        obs_b.append(obs_boot_i)
        sampleTime_b.append(unique_i)
        uniqueTime_b.append(unique_i)
    
            # This line runs differential evolution on the bootstraps
    optimized_b = differential_evolution(
        lambda params: CostFunction(params, params_trans, ODE_model, init_cond,
                               obs_b, sampleTime_b, uniqueTime_b,
                               ODE_names, return_all=False),
        bounds=bounds,
        maxiter=maxiter,
        tol=tol)

    # This line calls the CostFunction (defined above) and pulls out the total and individual RMSLEs
    RMSLE_boot_results = CostFunction(optimized_b.x, params_trans,
                                     ODE_model, init_cond,
                                     obs_b, sampleTime_b, uniqueTime_b,
                                     ODE_names, return_all=rmsle_return)

    return optimized_b.x, RMSLE_boot_results

# ---------------------------------------------------------------------------------#

#####################################################################
# Defining differential evolution run for full datset and bootstrap #
#####################################################################

def DE_Generalized(params_trans, ODE_model, init_cond, obs_data, sample_times, unique_times, bounds, n_boot=0, rmsle_return=False, maxiter=10000, tol=1e-5, plot_cost_hist=True, show_usage=False, ODE_names=None, n_jobs=1):

    # Error checks!
    # init_cond, obs_data, sample_times, unique_times, and bounds need to be passed as lists
    if not isinstance(init_cond, list):
        raise TypeError(f"{init_cond} needs to be passed as a list, even if there is only one initial condition. For example:\n "
                        "One initial condition -> ['Initial Condition 1'] \n" 
                        "More than one initial condition -> ['Initial Condition 1', 'Initial Condition 2', etc.]")

    if not isinstance(obs_data, list):
        raise TypeError(f"{obs_data} needs to be passed as a list, even if there is only one observed data column. For example:\n "
                        "One variable -> ['Variable A'] \n" 
                        "More than one variable -> ['Variable A', 'Variable B', etc.]")

    if not isinstance(sample_times, list):
        raise TypeError(f"{sample_times} needs to be passed as a list, even if there is only one sample time column. For example:\n "
                        "Sample times for one variable -> ['Sample Times for A'] \n" 
                        "Sample times for more than one variable -> ['Sample Times for A', 'Sample Times for B', etc.]")

    if not isinstance(unique_times, list):
        raise TypeError(f"{unique_times} needs to be passed as a list, even if there is only one unique time column. For example:\n "
                        "Unique times for one variable -> ['Unique Times for A'] \n" 
                        "Unique times for more than one variable -> ['Unique Times for A', 'Unique Times for B', etc.]")

    if not isinstance(bounds, list):
        raise TypeError(f"{bounds} needs to be passed as a list, even if there is only one parameter bound. For example:\n "
                        "Parameter bounds for one parameter -> [Parameter A Bounds] \n" 
                        "Parameter bounds for multiple parameters -> [Parameter A Bounds, Parameter B Bounds, etc.]")

    # These lines create a function to store the cost history of each model
    cost_hist_full = []

    def cost_record_full(xk, convergence):
        full_cost = CostFunction(xk, params_trans, ODE_model, init_cond,
                                obs_data, sample_times, unique_times,
                                ODE_names, return_all=False)
        cost_hist_full.append(full_cost)
        return False
    
    # This line runs the full data set through the differential evolution package
    optimized_full = differential_evolution(
        lambda params: CostFunction(params, params_trans, ODE_model, init_cond,
                               obs_data, sample_times, unique_times,
                               ODE_names, return_all=False),
        bounds=bounds,
        maxiter=maxiter,
        tol=tol,
        callback=cost_record_full)

    # This line pulls out the optimized parameters
    optParams_full = optimized_full.x

    # This line calls the CostFunction (defined above) and pulls out the total and individual RMSLEs
    RMSLE_Results = CostFunction(optParams_full, params_trans,
                                           ODE_model, init_cond,
                                           obs_data, sample_times, unique_times,
                                           ODE_names, return_all=rmsle_return)

    # These lines either return both the total and individual RMSLEs or just the total RMSLE depending on what the user specified
    if rmsle_return:
        Total_RMSLE_Full, RMSLEs_Full = RMSLE_Results
    else:
        Total_RMSLE_Full = RMSLE_Results
        RMSLEs_Full = None

    # This line provides a usage guide for calling the correct results for future graphing and analysis
    if show_usage:
        print("\n" + "="*60)
        print("Guide to Pulling Individual Results from `DE_Generalized`")
        print("="*60)

        print("\nOrder of outputs corresponds to order of ODE equations in ODE_model and order of parameter bounds entered:")

        if rmsle_return:
            if n_boot == 0:
                print("\nOptimal Parameters (Full Model):")
                print("  results[0]")
    
                print("\nTotal RMSLE (Full Model):")
                print("  results[1][0]")
    
                print("\nIndividual RMSLEs by ODE (Full Model):")
                print("  results[1][1]")
            else:
                print("\nOptimal Parameters (Full Model):")
                print("  results[0]")
    
                print("\nTotal RMSLE (Full Model):")
                print("  results[1][0]")
    
                print("\nIndividual RMSLEs by ODE (Full Model):")
                print("  results[1][1]")
                
                print("\nOptimal Parameters (Bootstrap):")
                print("  results[2]")
    
                print("\nTotal RMSLE (Bootstrap):")
                print("  results[3]")

                print("\nTotal RMSLE (Bootstrap):")
                print("  results[3]")

        else:
            if n_boot == 0:
                print("\nOptimal Parameters (Full Model):")
                print("  results[0]")
    
                print("\nTotal RMSLE (Full Model):")
                print("  results[1][0]")
    
            else:
                print("\nOptimal Parameters (Full Model):")
                print("  results[0]")
    
                print("\nTotal RMSLE (Full Model):")
                print("  results[1][0]")
                
                print("\nOptimal Parameters (Bootstrap):")
                print("  results[2]")
    
                print("\nTotal RMSLE (Bootstrap):")
                print("  results[3]")

                print("\nTotal RMSLE (Bootstrap):")
                print("  results[3]")

        print("="*60 + "\n")
        
    # This line specifies to stop after running the full model if the number of bootstraps is set to 0 in the argument line
    if n_boot == 0:
        return (optParams_full, (Total_RMSLE_Full, RMSLEs_Full))
        
    else:
        # These lines set up the bootstrapping by creating empty lists and calculating the number of parameters we want to estimate

        bootstrap_loop_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(boot_run)(
                b, params_trans, ODE_model, init_cond, obs_data, sample_times, unique_times, bounds, maxiter, tol, rmsle_return, ODE_names)
            for b in range(n_boot))
        
        n_params = len(bounds)
        boot_params = np.zeros((n_boot, n_params))
        boot_RMSLE_Total = np.zeros(n_boot)
        boot_RMSLEs = []

        for b, (params_b, rmsle_b) in enumerate(bootstrap_loop_results):
            boot_params[b, :] = params_b

            if rmsle_return:
                Total_RMSLE_b, RMSLEs_b = rmsle_b
            else:
                Total_RMSLE_b = rmsle_b
                RMSLEs_b = None

            boot_RMSLE_Total[b] = Total_RMSLE_b
            boot_RMSLEs.append(RMSLEs_b)

        # Plotting cost histories for full model/bootstraps
        if plot_cost_hist:
            plt.figure(figsize=(6,4))
            plt.plot(cost_hist_full, lw=2)
            plt.xlabel('Iteration')
            plt.ylabel('Total RMSLE')
            plt.title('Cost History (Full Model)')
            plt.grid(False)
            plt.show()
            
        return (optParams_full, (Total_RMSLE_Full, RMSLEs_Full), boot_params, boot_RMSLE_Total, boot_RMSLEs)

# ---------------------------------------------------------------------------------#

#################################################################
# Defining full and bootstrap model outputs from DE_generalized #
#################################################################

def DE_Results(DE_output, params_trans, init_cond, unique_times, ODE_model, conf_int=(2.5, 97.5), show_usage=False, param_names=None):

    # These lines pull out the optimized paramters for both the full model and the bootstraps
    optParams_full = DE_output[0]
    
    optParams_boot = np.asarray(DE_output[2])
    n_boot, n_params = optParams_boot.shape

    # These lines undo log_10 transformations if specified by the user, otherwise, the original parameter remains unchanged (linear line)
    # This line creates an empty list to store un-transformed/original parameters
    transformed_params_full = []
    transformed_params_boot = []

    # This for loop works through parameters specified by user to undo log_10 transforms or leave parameters unchanged (as specified by the user).
    # Specification for transformations, or lack thereof, should be entered as a list: params_trans = ['linear', 'log10', 'log10', 'linear', etc.]
    # NOTE: Parameter transformation specifications must match the exact order parameters are specified in. So, if you have parameters a, b, c, and d,
    #       you enter the parameters as params = ['a', 'b', 'c', 'd'], and you log-transformed b and c, then you would need to specify the
    #       order for params_trans as: params_trans = ['linear', 'log10', 'log10', 'linear'] to un-log transform b and c for the ODE solver.
    for value_full, rule in zip(optParams_full, params_trans):
        if rule == "linear":
            transformed_params_full.append(value_full)
        elif rule == "log10":
            transformed_params_full.append(10**value_full)
        else:
            raise ValueError(f"In your transform list '{rule}', you entered a transform that is not supported, or you misspecified a transform that is supported. The only transforms supported at this time are 'linear' (which leaves your parameter unchanged) or 'log10' (which un-logs a parameter you might have had on the log10 scale). Also, since you are getting this error, double-check that the parameter order you specified matches the same transform order you need :)")

    for b in range(n_boot):
        boot_param_vec = []
        for value_boot, rule in zip(optParams_boot[b], params_trans):
            if rule == "linear":
                boot_param_vec.append(value_boot)
            elif rule == "log10":
                boot_param_vec.append(10**value_boot)
            else:
                raise ValueError(f"In your transform list '{rule}', you entered a transform that is not supported, or you misspecified a transform that is supported. The only transforms supported at this time are 'linear' (which leaves your parameter unchanged) or 'log10' (which un-logs a parameter you might have had on the log10 scale). Also, since you are getting this error, double-check that the parameter order you specified matches the same transform order you need :)")
        transformed_params_boot.append(boot_param_vec)

    # These lines put the full and bootstrapped model parameters into a usable form for computing fitted lines and confidence intervals
    transformed_params_full = tuple(transformed_params_full)
    transformed_params_boot = np.asarray(transformed_params_boot)

    # This line compute the end time from the unique time columns
    end_time = max(np.max(max_time) for max_time in unique_times)
    
    # This line creates a list of times to create the fitted line
    t_list = np.linspace(0, end_time, 1000)

    # This line creates the data for the fitted line(s) of the ODE system for the full model
    ODE_full = odeint(ODE_model, init_cond, t_list, args=tuple(transformed_params_full))

    # These lines create the data for the confidence interaval of the ODE system from the bootstraps
    n_ODE_sol = ODE_full.shape[1]
    
    boot_fit = np.zeros((n_boot, len(t_list), n_ODE_sol))
    
    for i in range(n_boot):
        boot_params_i = transformed_params_boot[i]
        ODE_boot_i = odeint(ODE_model, init_cond, t_list, args=tuple(boot_params_i))
        boot_fit[i] = ODE_boot_i

    # These lines compute the upper and lower bounds of the confidence interval for the fit
    CI_lower = np.percentile(boot_fit, conf_int[0], axis=0)
    CI_upper = np.percentile(boot_fit, conf_int[1], axis=0)

    CI_l_params = np.percentile(transformed_params_boot, conf_int[0], axis=0)
    CI_u_params = np.percentile(transformed_params_boot, conf_int[1], axis=0)

    # These lines print out the usage of the function in order to pull results for graphing
    if show_usage:
        print("\n" + "="*50)
        print("Guide to Pulling Individual Results from `DE_Results`")
        print("="*50)

        print("NOTE: Order of ODE solutions matches order specified by user in ODE_model")
        
        print("\nTime:")
        print("  results['time']")
        
        print("\nFull ODE Solutions:")
        print("  results['Full ODE Solutions'][:, j]")
        print("    where j = index of individual ODE solution")
        
        print("\nBootstrap ODE Solutions:")
        print("  results['Bootstrap ODE Solutions'][:, :, j]")
        print("    where j = index of individual ODE solution")
        
        print("\nLower Confidence Interval for Fit:")
        print("  results['Lower CI: Fit'][:, j]")
        print("    where j = index of individual ODE solution")
        
        print("\nUpper Confidence Interval for Fit:")
        print("  results['Upper CI: Fit'][:, j]")
        print("    where j = index of individual ODE solution")
        
        print("\nFull ODE Parameter Estimates:")
        print("  results['Full ODE Parameter Estimates']")
        print("\nBootstrap ODE Parameter Estimates:")
        print("  results['Bootstrap ODE Parameter Estimates']")
        
        print("\nLower Confidence Interval for Parameters:")
        print("  results['Lower CI: Parameters'][:, k]")
        print("    where k = index of individual parameter estimate")
        
        print("\nUpper Confidence Interval for Parameters:")
        print("  results['Upper CI: Parameters'][:, k]")
        print("    where k = index of individual parameter estimate")

        print("\n Table of Parameter Estimates and CIs:")
        print("   results['Parameter Table']")
            
        print("="*50 + "\n")

    if param_names is None:
        param_names = [f"param_{i+1}" for i in range(len(params_trans))]

    param_est_df = pd.DataFrame({"Parameter": param_names,
                                "Estimate (Full Model)": transformed_params_full,
                                "Lower CI": CI_l_params,
                                "Upper CI": CI_u_params})

    return {"time": t_list,
          "Full ODE Solutions": ODE_full,
          "Bootstrap ODE Solutions": boot_fit,
          "Lower CI: Fit": CI_lower,
          "Upper CI: Fit": CI_upper,
          "Full ODE Parameter Estimates": transformed_params_full,
          "Bootstrap ODE Parameter Estimates": transformed_params_boot,
           "Lower CI: Parameters": CI_l_params,
           "Upper CI: Parameters": CI_u_params,
           "Parameter Table": param_est_df}

# ---------------------------------------------------------------------------------#

####################################################
# Defining likelihood profile function (HOPEFULLY) #
####################################################






