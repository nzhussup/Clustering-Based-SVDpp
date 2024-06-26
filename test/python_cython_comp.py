# THIS IS THE COMPARISON SCRIPT OF CYTHON AND PURE PYTHON VERSIONS OF MY IMPLEMENTATIONS
## WE WILL COMPARE THE RUNTIMES ON 5 EPOCHS

import cython_cb_svdpp
import pure_implementation
import cluster_rec
from surprise import Dataset
from surprise.model_selection import train_test_split
from time import time

cython_algo = cython_cb_svdpp.CB_SVDpp(n_epochs=5)
pure_algo = pure_implementation.CB_SVDpp(n_epochs=5)
surprise_algo = cluster_rec.CB_SVDpp(n_epochs=5, num_clusters=50, alpha=0.15)
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)


def calc_time(model, name: str, surprise=False):
    t1 = time()
    print(f"Processing {name}")
    model.fit(trainset)
    t2 = time()
    print(f"Took {round(t2-t1,2)} seconds to fit.")
    t1 = time()
    if surprise:
        model.predict_df(testset)
    else:
        model.predict(testset)
    t2 = time()
    print(f"Took {round(t2-t1, 2)} seconds to predict.")
    
calc_time(pure_algo, "pure python implementation."
calc_time(cython_algo, "cython implementation.")
calc_time(surprise_algo, "surprise based implementation.", surprise=True)