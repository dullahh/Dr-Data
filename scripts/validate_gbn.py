import numpy as np
import pandas as pd
from networkx import topological_sort
from scipy.stats import norm

from data_pipeline import fetch_data, build_raw_dataframe, bn_df_construction
from model_gbn import DAG_EDGES, fit_model


TARGET = "Asset_2_Return"
CONFIDENCE_LEVELS = (0.95, 0.99)



def validate_input_data(data):
    required_columns = {node for edge in DAG_EDGES for node in edge}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing model columns: {sorted(missing_columns)}")

    model_data = data[list(required_columns)]

    if model_data.empty:
        raise ValueError("No data available.")

    if model_data.isna().any().any():
        raise ValueError("Missing values found.")

    if not np.isfinite(model_data.to_numpy()).all():
        raise ValueError("Infinite values found.")


def split_time_series(data: pd.DataFrame, test_fraction: float):
    
    split_at = int(len(data) * (1 - test_fraction))
    
    return data.iloc[:split_at].copy(), data.iloc[split_at:].copy()


def conditional_target_distribution(model, observed_data: pd.DataFrame):
    """
        pgmpy solves a system using this covariance matrix
    """
   

    ordered_nodes = list(topological_sort(model))
    observed_nodes = [node for node in ordered_nodes if node in observed_data.columns]
    observed_indices = [ordered_nodes.index(node) for node in observed_nodes]
    _, joint_covariance = model.to_joint_gaussian()
    observed_covariance = joint_covariance[np.ix_(observed_indices, observed_indices)]
    if np.linalg.matrix_rank(observed_covariance) < len(observed_nodes):
        raise ValueError(
            "Cannot predict: the observed covariance matrix is singular. "
            
        )

    missing_variables, means, covariance = model.predict_probability(observed_data)
    target_index = missing_variables.index(TARGET)
    variance = float(covariance[target_index, target_index])
    if variance <= 0:
        raise ValueError(f"Conditional variance for {TARGET} is not positive: {variance}")
    return means[:, target_index], float(np.sqrt(variance))


def print_risk_metrics(actual: pd.Series, mean: np.ndarray, std: float):
    """Back-test (from 1 side) VaR as a return threshold"""
    for confidence in CONFIDENCE_LEVELS:
        lower_return_threshold = mean + norm.ppf(1 - confidence) * std
        breaches = actual.to_numpy() < lower_return_threshold
        expected_breach_rate = 1 - confidence
        print(
            f"{confidence} VaR return threshold: mean {lower_return_threshold.mean():.4%}; "
            f"breaches {breaches.mean():.2%} (expected {expected_breach_rate:.2%})"
        )


def validate_model(data: pd.DataFrame, test_fraction: float):
    validate_input_data(data)
    train, test = split_time_series(data, test_fraction)
    required_columns = {node for edge in DAG_EDGES for node in edge}
    constant_columns = [column for column in sorted(required_columns) if train[column].nunique() < 2]
    if constant_columns:
        raise ValueError(
            f"Training data has constant model columns: {constant_columns}. "
            "These need historical values or must be removed from the DAG."
        )
    model = fit_model(train)

    """
    pgmpy's built-in structural check: CPD parent sets agree with the DAG.
    """
    model.check_model()
    if len(model.get_cpds()) != len(model.nodes()):
        raise ValueError("Not every model node has a fitted CPD.")

    observed_test = test.drop(columns=TARGET)
    predicted_mean, predicted_std = conditional_target_distribution(model, observed_test)
    actual = test[TARGET]
    residuals = actual.to_numpy() - predicted_mean
    naive_residuals = actual.to_numpy() - train[TARGET].mean()


    test_log_likelihood = model.log_likelihood(test)
    print(f"Rows: train={len(train)}, holdout={len(test)}")
    print("Structure: valid CPD-to-parent mapping")
    print(f"Holdout log-likelihood per row: {test_log_likelihood / len(test):.4f}")
    print(f"Target: {TARGET}")
    print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.4%}")
    print(f"Naive-mean RMSE: {np.sqrt(np.mean(naive_residuals**2)):.4%}")
    print(f"Direction accuracy: {np.mean(np.sign(predicted_mean) == np.sign(actual)):.4%}")
    print(f"Conditional target volatility: {predicted_std:.4%}")
    print_risk_metrics(actual, predicted_mean, predicted_std)


def main():

    datasets = fetch_data(days_window=100, yf_period="100d")
    df_raw = build_raw_dataframe(datasets)
    df_candidates = bn_df_construction(df_raw)

    model_columns = sorted({node for edge in DAG_EDGES for node in edge})
    df_bn = df_candidates.loc[:, model_columns].sort_index().dropna()

    validate_model(df_bn, 0.20)


if __name__ == "__main__":
    main()
