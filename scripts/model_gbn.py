from pgmpy.models import LinearGaussianBayesianNetwork
import numpy as np

# DAG_EDGES = [
#     ('Yield_10Y', 'XLK_Return'),
#     ('Sentiment', 'VIX_Return'),
#     ('VIX_Return', 'IV_Change'),
#     ('XLK_Return', 'Asset_1_Return'),
#     ('XLK_Return', 'Asset_2_Return'),
#     ('IV_Change', 'Asset_1_Return'),
#     ('Asset_1_Return', 'Asset_2_Return')
# ]

DAG_EDGES = [
    ("Yield_10Y", "XLK_Return"),
    ("XLK_Return", "VIX_Return"),
    ("XLK_Return", "Asset_1_Return"),
    ("VIX_Return", "Asset_1_Return"),
    ("XLK_Return", "Asset_2_Return"),
    ("Asset_1_Return", "Asset_2_Return"),
]

def fit_model(train_df):
    model = LinearGaussianBayesianNetwork(DAG_EDGES)
    train = train_df.loc[:, list(model.nodes())]

    if not np.isfinite(train.to_numpy(dtype=float)).all():
        raise ValueError("Training data contains missing or infinite values.")

    model.fit(train)
    model.check_model()
    return model


def print_parameters(model):
    for cpd in model.get_cpds():
        print(f"Node: {cpd.variable}")
        print(f"  Parents: {cpd.evidence}")
        print(f"  Intercept: {cpd.beta[0]}")
        print(f"  Parent coefficients: {cpd.beta[1:]}")
        print(f"  Residual variance: {cpd.std ** 2}\n")