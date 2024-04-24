import numpy as np
from surprise import Dataset
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pure_implementation import CB_SVDpp
from time import time

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)

t1 = time()
algo = CB_SVDpp(n_epochs=3, verbose=True)
algo.fit(trainset)
y_pred, y_true = algo.predict(testset)
t2 = time()

print(f"RMSE: {np.sqrt(mean_squared_error(y_pred, y_true))} in 3 epochs.")
print(f"Took {round(t2-t2, 2)} seconds for 3 epochs.") # Approx 110 seconds for epoch

