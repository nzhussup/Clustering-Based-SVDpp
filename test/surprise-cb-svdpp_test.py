import numpy as np
from surprise import Dataset
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from cluster_rec import CB_SVDpp
from time import time
from custom_cross_validation import custom_cross_validation

# Import and split data
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)

# Train on random values
algo = CB_SVDpp(num_clusters=50, alpha=0.15, n_epochs=20, verbose=True)
start = time()
algo.fit(trainset)
algo.calc_Nu(trainset)

# Predict on random values and calculate the RMSE and runtime
y_pred, y_true = algo.predict_df(testset)
end = time()
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse}")
print(f"Took {round(end-start,2)} seconds to fit and predict.")

# Initialize different alpha and cluster values
alpha_values = [0.1, 0.15, 0.2]
cluster_values = [50, 100, 150]

# Cross validate alpha and cluster values. Output best params and total runtime
cv_start = time()
results, best_params = custom_cross_validation(CB_SVDpp, [trainset, testset], alpha_values, cluster_values, cv=3, surprise=True, verbose=False)
cv_end = time()
print(f"CV total runtime: {round(cv_end - cv_start, 2)} seconds.")
