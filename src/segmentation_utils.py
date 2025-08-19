import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_rfm(df: pd.DataFrame, invoice_date_col: str = "InvoiceDate", amount_col: str = "Amount", customer_col: str = "CustomerID", reference_date: str = None) -> pd.DataFrame:
    '''
    Compute Recency, Frequency and Monetary metrics for each customer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing transaction data.
    invoice_date_col : str
        Name of the column containing transaction dates.
    amount_col : str
        Name of the column containing transaction amounts.
    customer_col : str
        Name of the column containing customer identifiers.
    reference_date : str or None
        The date used to compute recency (format: 'YYYY-MM-DD'). If None, uses the max invoice date.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by customer with Recency, Frequency and Monetary columns.
    '''
    # Convert dates to datetime
    df[invoice_date_col] = pd.to_datetime(df[invoice_date_col])
    if reference_date is None:
        reference_date = df[invoice_date_col].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)

    # Aggregate metrics
    rfm = df.groupby(customer_col).agg(
        {
            invoice_date_col: lambda x: (reference_date - x.max()).days,
            customer_col: "count",
            amount_col: "sum",
        }
    )
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    return rfm


def kmeans_rfm(rfm: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    '''
    Apply KMeans clustering to RFM data.

    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with R, F, M metrics.
    n_clusters : int
        Number of clusters to form.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    tuple
        The RFM DataFrame with an additional 'Cluster' column and the average silhouette score.
    '''
    # Normalise metrics
    rfm_normalised = (rfm - rfm.mean()) / rfm.std()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(rfm_normalised)
    rfm = rfm.copy()
    rfm["Cluster"] = cluster_labels
    # Compute silhouette
    if n_clusters > 1:
        score = silhouette_score(rfm_normalised, cluster_labels)
    else:
        score = float('nan')
    return rfm, score
