# THIS IS THE COMPARISON SCRIPT OF CYTHON AND PURE PYTHON VERSIONS OF MY IMPLEMENTATIONS
## WE WILL COMPARE THE RUNTIMES ON 5 EPOCHS

import cython_cb_svdpp
import pure_implementation
from surprise import Dataset
from surprise.model_selection import train_test_split
from time import time

cython_algo = cython_cb_svdpp.CB_SVDpp(n_epochs=3)
pure_algo = pure_implementation.CB_SVDpp(n_epochs=3, verbose=False)
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)


def calc_time(model, name: str):
    t1 = time()
    print(f"Processing {name}")
    model.fit(trainset)
    t2 = time()
    print(f"Took {round(t2-t1,2)} seconds to fit.")
    t1 = time()
    model.predict(testset)
    t2 = time()
    print(f"Took {round(t2-t1, 2)} seconds to predict.")
    
calc_time(pure_algo, "pure python implementation.")
calc_time(cython_algo, "cython implementation.")