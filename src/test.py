import cython_cb_svdpp
from surprise import Dataset
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

algo = cython_cb_svdpp.CB_SVDpp(n_epochs=1, verbose=True)
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)

algo.fit(trainset)
y_pred, y_true = algo.predict(testset)
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred))}")
