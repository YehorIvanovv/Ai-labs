import numpy as np
from sklearn import cluster
import warnings

warnings.filterwarnings("ignore")

names = np.array([
    "Apple", "Microsoft", "Google", "Amazon", "IBM",
    "ExxonMobil", "Chevron", "ConocoPhillips",
    "Walmart", "Target", "Costco"
])

np.random.seed(42)
days = 500

tech_trend = np.random.randn(days)
energy_trend = np.random.randn(days)
retail_trend = np.random.randn(days)

quotes_diff = []
for _ in range(5): quotes_diff.append(tech_trend + np.random.randn(days) * 0.5)
for _ in range(3): quotes_diff.append(energy_trend + np.random.randn(days) * 0.5)
for _ in range(3): quotes_diff.append(retail_trend + np.random.randn(days) * 0.5)

X = np.array(quotes_diff).astype(float).T

X /= X.std(axis=0)

try:
    from sklearn.covariance import GraphicalLassoCV
    edge_model = GraphicalLassoCV()
except ImportError:
    from sklearn.covariance import GraphLassoCV
    edge_model = GraphLassoCV()

with np.errstate(invalid='ignore'):
    edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
num_labels = labels.max()

print("Результати кластеризації компаній на фондовому ринку:\n")
for i in range(num_labels + 1):
    cluster_names = names[labels == i]
    if len(cluster_names) > 0:
        print(f"Кластер {i+1} ==> {', '.join(cluster_names)}")
