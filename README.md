# A General ODE Bootstrap
A package to perform parameter estimation for ordinary differential equations (ODEs) using the Differential Evolution (DE) algorithm, with an option to perform bootstrapping and plot profile likelihoods. There is also an option to run bootstrapping on multiple CPU processors.

## Requirements
For importing into your python workspace:
  - numpy
  - pandas
  - odeint from scipy.integrate
  - differential_evolution from scipy.optimize
  - Counter from collections
  - matplotlib

For running functions within this package:
  - ODE model (you need to pre-specify your ODE model)
  - Time series data (this package was written for ODE models dealing with data that has a time component)
  - Data from different treatment groups needs to be separated out before running anything
  - Initial conditions for all ODE equations

## Using this Package
1. Import all required libraries and packages:
```python
from A_General_ODE_Bootstrap import timeseries_pull, observedData_pull, CostFunction, DE_Generalized, DE_Results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
import seaborn as sns
from collections import Counter
from matplotlib.ticker import MultipleLocator
```
2. Import data as a `pandas` dataframe:
```python
your_data = pd.read_csv('your_data_file.csv')
# OR
your_data = pd.read_excel('your_data_file.xlsx')
```
Doing this should result in a dataframe that looks, in some way, shape, or form, like:
| Time | Variable A | Variable B | $$\cdots$$ | Variable $k$ |
|-------|------------|------------|------------|--------------|
| $t_0$ | $$A_0$$    | $$B_0$$    | $$\cdots$$ | $$k_0$$      |
| $t_1$ | $$A_1$$    | $$B_1$$    | $$\cdots$$ | $$k_1$$      |
| $t_2$ | $$A_2$$    | $$B_2$$    | $$\cdots$$ | $$k_2$$      |
| $\vdots$ | $\vdots$ |$\vdots$   | $$\ddots$$ | $\vdots$     |
| $t_n$ | $$A_n$$    | $$B_n$$    | $$\cdots$$ | $$k_n$$      |

Or, if you have more than one treatment group in your dataframe:
| Time | Treatment |Variable A | Variable B | $$\cdots$$ | Variable $k$ |
|------|-----------|-----------|------------|------------|--------------|
| $t_0$ | $\alpha$ | $$A_0$$    | $$B_0$$    | $$\cdots$$ | $$k_0$$     |
| $t_1$ | $\alpha$ | $$A_1$$    | $$B_1$$    | $$\cdots$$ | $$k_1$$     |
| $t_2$ | $\alpha$ | $$A_2$$    | $$B_2$$    | $$\cdots$$ | $$k_2$$     |
| $\vdots$ | $\alpha$ | $\vdots$ |$\vdots$   | $$\ddots$$ | $\vdots$       |
| $t_n$ | $\alpha$ | $$A_n$$    | $$B_n$$    | $$\cdots$$ | $$k_n$$     |
| $t_0$ | $\beta$  | $$A_0$$    | $$B_0$$    | $$\cdots$$ | $$k_0$$     |
| $t_1$ | $\beta$  | $$A_1$$    | $$B_1$$    | $$\cdots$$ | $$k_1$$     |
| $t_2$ | $\beta$  | $$A_2$$    | $$B_2$$    | $$\cdots$$ | $$k_2$$     |
| $\vdots$ | $\beta$ | $\vdots$ |$\vdots$   | $$\ddots$$ | $\vdots$       |
| $t_n$ | $\beta$  | $$A_n$$    | $$B_n$$    | $$\cdots$$ | $$k_n$$     |

3. If there are multiple treatment groups, separate dataframe by treatment group:

| Time | Treatment |Variable A | Variable B | $$\cdots$$ | Variable $k$ |
|------|-----------|-----------|------------|------------|--------------|
| $t_0$ | $\alpha$ | $$A_0$$    | $$B_0$$    | $$\cdots$$ | $$k_0$$     |
| $t_1$ | $\alpha$ | $$A_1$$    | $$B_1$$    | $$\cdots$$ | $$k_1$$     |
| $t_2$ | $\alpha$ | $$A_2$$    | $$B_2$$    | $$\cdots$$ | $$k_2$$     |
| $\vdots$ | $\alpha$ | $\vdots$ |$\vdots$   | $$\ddots$$ | $\vdots$    |
| $t_n$ | $\alpha$ | $$A_n$$    | $$B_n$$    | $$\cdots$$ | $$k_n$$     |

| Time | Treatment |Variable A | Variable B | $$\cdots$$ | Variable $k$ |
|------|-----------|-----------|------------|------------|--------------|
| $t_0$ | $\beta$  | $$A_0$$    | $$B_0$$    | $$\cdots$$ | $$k_0$$     |
| $t_1$ | $\beta$  | $$A_1$$    | $$B_1$$    | $$\cdots$$ | $$k_1$$     |
| $t_2$ | $\beta$  | $$A_2$$    | $$B_2$$    | $$\cdots$$ | $$k_2$$     |
| $\vdots$ | $\beta$ | $\vdots$ |$\vdots$    | $$\ddots$$ | $\vdots$    |
| $t_n$ | $\beta$  | $$A_n$$    | $$B_n$$    | $$\cdots$$ | $$k_n$$     |

4. Define ODE model:
```python
def ODE_model(y, t, parameter 1, parameter 2, etc.):
  # Define all ODE variables
  A, B = y

  # Define any fixed constants that belong in the ODE equation that will NOT be solved for
  c = # some constant

  # Define ODE equations
  dA = # some equation
  dB = # some equation

  # Put ODE equations in a list
  dy = [dA, dB]

  return dy
```
NOTE: The order in which you return your ODEs is the same order in which solutions will be returned when solving the ODEs using Differential Evolution and the ODE solver.

5. Pull time series information using `timeseries_pull(df, time_column)`:
```python
sample_t, unique_t = timeseries_pull(your_data, 'your time column label')

# OR (if multiple treatment groups)

sample_t_a, unique_t_a = timeseries_pull(your_data_trt_a, 'your time column label')
sample_t_b, unique_t_b = timeseries_pull(your_data_trt_b, 'your time column label')
```
NOTE: To pull the correct time series data, the column name entered in the function where time is stored needs to match what is in your dataframe.

6. Pull observed data using `observedData_pull(df, data_columns)`:
```python
your_obs = observedData_pull(your_data, ['Variable A', 'Variable B', etc.])

# OR (if multiple treatment groups)

your_obs_trt_a = observedData_pull(your_data_trt_a, ['Variable A', 'Variable B', etc.])
your_obs_trt_b = observedData_pull(your_data_trt_b, ['Variable A', 'Variable B', etc.])
```
NOTE: Observed data columns need to be entered as a list into `observedData_pull`, even if there is only one variable in the dataframe.

7. Define parameter bounds for Differential Evolution to search through:
```python
parameter 1 = lower bound, upper bound
parameter 2 = lower bound, upper bound

param_bounds = [parameter 1, parameter 2]
```
8. Use `DE_Generalized` to solve system of ODEs, and implement optional bootstrapping:
```python
de_gen_output = DE_Generalized(['linear', 'log10'], # List of transforms to undo if parameters where transformed: 'Linear' -> unchanged
                                                    #                                                             'log10' -> undo log10 transformation of parameter

                               ODE_model, # ODE model defined earlier

                               [A_0, B_0], # Initial conditions for each ODE equation
                                           # NOTE: Order of initial conditions needs to match order of ODE equations in ODE_model

                               [your_obs[0], your_obs[1]], # Observed data pulled from observedData_pull: [0] -> pulls observed data for first variable
                                                           #                                              [1] -> pulls observed data for second variable
                                                           # NOTE: Order of observed variables needs to match order of ODE equations in ODE_model

                               [sample_t_A, sample_t_B], # Sample times pulled from timeseries_pull
                                                         # NOTE: If the sample time is the same for all variables, the same sample time output from timeseries_pull can be used mutliple times for as many variables as are observed

                               [unique_t_A, unique_t_B], # Unique times pulled from timeseries_pull
                                                         # NOTE: If the unique time is the same for all variables, the same unique time output from timeseries_pull can be used mutliple times for as many variables as are observed

                               param_bounds, # Parameter bounds defined above for DE to search for parameter estimates

                               n_boot=0, # Number of bootstraps to be performed
                                         # NOTE: Change 0 to the number of bootstraps to be performed if bootstrapping is wanted

                               rmsle_return=True, # Returns individual RMSLEs for each ODE equation

                               maxiter=10000, # Number of iterations DE works for

                               tol=1e-5, # Error tolerance

                               plot_cost_history=True, # Returns plots of the cost history for the full model (no bootstraps) and, if n_boot > 0, plot of cost history of bootstraps

                               show_usage=False, # Provides more detailed information on how to pull individual results from DE_Generalized

                               ODE_names=None, # Provides a way to attach ODE name to individual RMSLEs upon return

                               n_jobs=1) # Specifies how many CPU processors bootstrapping should utilize
```
NOTE: `os.cpu_count()` can be used to determine exactly how many CPU processors are availables for use. More processors used results in reduced run-time for bootstrapping. Default value for the number of processors to be used is set to 1.

NOTE: If there are multiple treatment groups, you need to run `DE_Generalized` for each treatment group, separately.

9. Use `DE_Results` to get fitted line using estimated parameters, confidence intervals for parameters from bootstrapping, and confidence interval band from model fits from bootstrapping results:
```python
de_res_output = DE_Results(de_gen_output,

                           ['linear', 'log10'],

                           [A_0, B_0],

                           [unique_t_A, unique_t_B],

                           ODE_model,

                           (2.5, 97.5) # Confidence interval

                           show_usage=False,

                           param_names=['parameter 1', 'parameter 2']) # List of parameter names to produce a labeled table with parameter estimates and confidence intervals
```
10. View results:
```python
de_res_output['Parameter Table']
```
| Parameter | Estimate (Full Model) | Lower CI | Upper CI |
|-----------|-----------------------|----------|----------|
| parameter 1 | some estimated value | lower CI value | upper CI value |
| parameter 2 | some estimated value | lower CI value | upper CI value |

NOTE: Lower and Upper CI estimates come from bootstrapping.

11. Graph results:

Results can now be graphed. To pull individual results from the output of `DE_Results`:
```python
de_res_output['time'] # Time

de_res_output['Full ODE Solutions'][:, j] # Predicted values from full model parameter estimates
                                          # j is the index for each individual ODE solution

de_res_output['Lower CI: Fit'][:, j] # Lower bound of confidence interval for bootstrap fits
                                     # j is the index for each individual ODE solution

de_res_output['Upper CI: Fit'][:, j] # Upper bound of confidence interval for bootstrap fits
                                     # j is the index for each individual ODE solution
```
NOTE: Order of outputs for each ODE in `DE_Results` matches the same order ODE equations are defined and returned in from ODE_model.


