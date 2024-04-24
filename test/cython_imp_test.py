import cython_cb_svdpp
from surprise import Dataset
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from time import time
from custom_cross_validation import custom_cross_validation


algo = cython_cb_svdpp.CB_SVDpp(n_epochs=20, verbose=True)
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)

start = time()
algo.fit(trainset)
y_pred, y_true = algo.predict(testset)
end = time()
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred))}")
print(f"Took {round(end-start, 2)} seconds for 20 epochs.")

# Initialize different alpha and cluster values
alpha_values = [0.1, 0.15, 0.2]
cluster_values = [50, 100, 150]

# Cross validate alpha and cluster values. Output best params and total runtime
cv_start = time()
results, best_params = custom_cross_validation(cython_cb_svdpp.CB_SVDpp, [trainset, testset], alpha_values, cluster_values, cv=3, surprise=False, verbose=True)
cv_end = time()
print(f"CV total runtime: {round(cv_end - cv_start, 2)} seconds.")