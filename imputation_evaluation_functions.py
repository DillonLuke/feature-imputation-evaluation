import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

def get_data(sample_size: int, correlation_matrix: np.ndarray, random_seed=None):
    np.random.seed(random_seed)
    
    L = np.linalg.cholesky(correlation_matrix)
    
    random_data = np.random.normal(loc=0,
                                   scale=1,
                                   size=(len(correlation_matrix), sample_size))
    
    return np.transpose(L @ random_data)


def get_mcar(original_column: pd.Series, frac: float, random_state=None):
    mcar_column = original_column.copy()
    
    idx = mcar_column.sample(frac=frac, random_state=random_state).index
    
    mcar_column[idx] = np.nan
    
    return mcar_column


def get_imputations(X: pd.DataFrame, missing_column: str, random_state=None):
    imputation_df = pd.DataFrame(index=X.index)
    
    imputers = [
        ("mean", SimpleImputer()),
        ("median", SimpleImputer(strategy="median")),
        ("knn", KNNImputer()), 
        ("iter", IterativeImputer(random_state=random_state))
    ]
    
    for name, imputer in imputers:
        imp_data = imputer.fit_transform(X)[:, X.columns == missing_column]
        imputation_df[f"{missing_column}_{name}"] = imp_data
    
    return imputation_df


def get_imputation_results(data: pd.DataFrame, original_column: str, missing_columns, 
                       target_column: str):
    columns = [original_column] + list(missing_columns)
    
    functions = ["mean", "median", "std", lambda x: x.corr(data[target_column])]
    
    descriptives = (data[columns]
                    .agg(functions)
                    .rename(index={'<lambda>': 'correlation'}))
    
    errors = (descriptives[columns]
              .apply(lambda x: x - descriptives[original_column]))
    
    return (descriptives, errors)


def missingness_simulation(data: pd.DataFrame, original_column: str, imputation_columns, 
                           target_column: str, frac: float, random_state=None):
    results = data.copy()
    
    results[original_column+"_na"] = get_mcar(
        original_column=data[original_column], 
        frac=frac,
        random_state=random_state
    )
    
    imp_df = get_imputations(
        X=results[[original_column+"_na"] + list(imputation_columns)], 
        missing_column=original_column+"_na",
        random_state=random_state
    )
    
    results[imp_df.columns] = imp_df
    
    descriptives, errors = get_imputation_results(
        data=results, 
        original_column=original_column,
        missing_columns=imp_df.columns.tolist(),
        target_column=target_column
    )
    
    return descriptives, errors


def iterated_missingness_simulation(k: int, **kwargs):
    iteration_results = []
    aggregated_results = []
    
    for i in range(k):
        kwargs["random_state"] = i
        iteration_results.append(missingness_simulation(**kwargs))
    
    for result_type in zip(*iteration_results): # iterate through descriptives and errors
        aggregated_data = pd.DataFrame(
            data=np.mean(result_type, axis=0),
            columns=result_type[-1].columns,
            index=result_type[-1].index
        )

        aggregated_results.append(aggregated_data)
    
    return aggregated_results


def gridsearch_missingness_simulation(parameter_name, parameters, parameter_keys,
                                      k: int, **kwargs):
    results_by_parameter = []
    for parameter in parameters:
        kwargs[parameter_name] = parameter
        
        results_by_parameter.append(iterated_missingness_simulation(k=k, **kwargs))
    
    results_by_type = []
    for result_type in zip(*results_by_parameter):
        results_by_type.append(pd.concat(result_type, keys=parameter_keys, axis=0))
    
    return results_by_type


def plot_histograms(data: pd.DataFrame, columns, **kwargs):
    rows = np.ceil(len(columns) / 3).astype(int)
    
    fig = plt.figure(figsize=(12, rows*3))
    axs = [fig.add_subplot(rows, 3, i+1) for i in range(len(columns))]
    
    for ax, col in zip(axs, columns):
        sns.histplot(data=data, x=col, ax=ax, **kwargs)
        
        
def plot_scatterplots(data: pd.DataFrame, columns, target_column, **kwargs):
    rows = np.ceil(len(columns) / 3).astype(int)
    
    fig = plt.figure(figsize=(12, rows*3))
    axs = [fig.add_subplot(rows, 3, i+1) for i in range(len(columns))]
    
    for ax, col in zip(axs, columns):
        sns.scatterplot(data=data, x=col, y=target_column, ax=ax, **kwargs)
        

def missingness_simulation_plot(iterated_results: pd.DataFrame, **kwargs):
    results_for_plot = (iterated_results
                        .stack()
                        .reset_index()
                        .set_axis(["metric", "imputation", "value"], axis=1))

    sns.lineplot(data=results_for_plot, x="metric", y="value", hue="imputation", **kwargs)
    
    
def gridsearch_simulation_plot(gridsearch_results: pd.DataFrame, **kwargs):
    gs_results_for_plot = (gridsearch_results
                           .stack()
                           .reset_index()
                           .set_axis(["parameter", "metric", "imputation", "value"],
                                   axis=1))

    fg = sns.FacetGrid(data=gs_results_for_plot, col= "metric", col_wrap=2)

    fg.map_dataframe(sns.lineplot, x="parameter", y="value", hue="imputation", **kwargs)
    
    fg.add_legend()

