# Import necessary libraries
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt
from time import time
from sklearn.cluster import KMeans
from collections import defaultdict

# Define CB_SVDpp class
cdef class CB_SVDpp:
    
    def __cinit__(self, int n_factors=20, int n_epochs=20, double init_mean=0, double init_std_dev=.1,
                double reg_param=0.1, double lr=0.01, int random_state=123, bint verbose=False,
                double alpha=0.15, int n_clusters=50):

        # Initialize attributes
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.n_factors = n_factors
        self.lr = lr
        self.reg_param = reg_param
        self.pu = None
        self.pCu = None
        self.u_labels = None
        self.qi = None
        self.qCi = None
        self.i_labels = None
        self.yj = None
        self.yCj = None
        self.y_labels = None
        self.bi = None
        self.bu = None
        self.Nu_minus_half = defaultdict(float)
        self.Nu = defaultdict(set)

    # Define method to calculate Nu
    def calc_Nu(self, trainset):
        for u, i, _ in trainset.all_ratings():
            self.Nu[u].add(i)
        for u in self.Nu:
            self.Nu_minus_half[u] = 1 / sqrt(len(self.Nu[u]))
        self.Nu = {u: list(items) for u, items in self.Nu.items()}

    # Define predictor method
    cdef double _predictor(self, int user_id, int item_id):
        cdef int j, cluster_id
        cdef double res
        cdef double[:] part1 = np.zeros(self.n_factors, dtype=np.float64)
        cdef double[:] part2 = np.zeros(self.n_factors, dtype=np.float64)
        cdef double[:] sum_result = np.zeros(self.n_factors, dtype=np.float64)

        part1 = (1 - self.alpha) * self.qi[:, item_id] + self.alpha * self.qCi[:, self.i_labels[item_id]]
        part2 = (1 - self.alpha) * self.pu[:, user_id] + self.alpha * self.pCu[:, self.u_labels[user_id]]

        for j in self.Nu[user_id]:
            cluster_id = self.i_labels[j]
            sum_result += (1 - self.alpha) * self.yj[:, j] + self.alpha * self.yCj[:, cluster_id]

        res = np.dot(part1, part2 + (self.Nu_minus_half[user_id] * sum_result)) + self.bi[item_id] + self.bu[user_id]
        return res

    # Define fit method
    def fit(self, trainset):
        
        self.trainset = trainset

        # Initialize latent factors and biases
        self.pu = np.random.normal(self.init_mean, self.init_std_dev, (trainset.n_users, self.n_factors))
        self.qi = np.random.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))
        self.yj = np.random.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))
        
        self.bu = np.zeros(trainset.n_users)
        self.bi = np.zeros(trainset.n_items)

        # Calculate Nu and Nu_minus_half
        self.calc_Nu(trainset)

        # Clustering
        kmeans_users = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans_items = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans_yj = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

        # Fit clusters
        self.u_labels = kmeans_users.fit_predict(self.pu)
        self.i_labels = kmeans_items.fit_predict(self.qi)
        self.y_labels = kmeans_yj.fit_predict(self.yj)

        # Initialize cluster centers
        self.pCu = kmeans_users.cluster_centers_
        self.qCi = kmeans_items.cluster_centers_
        self.yCj = kmeans_yj.cluster_centers_

        # SGD
        for _ in range(self.n_epochs):
            print(f"Processing epoch {_}")
            start = time()
            for u, i, r_ui in trainset.all_ratings():
                err = r_ui - self._predictor(u, i)
                # Update biases
                self.bu[u] += self.lr * (err - self.reg_param * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg_param * self.bi[i])

                # Save old values for updates
                pu_old = self.pu[u]
                qi_old = self.qi[i]

                # Update user and item factors
                self.pu[u] += self.lr * (err * qi_old - self.reg_param * self.pu[u])
                self.qi[i] += self.lr * (err * pu_old - self.reg_param * self.qi[i])

                # Update yj and yCj for all items j in Nu(u)
                for j in self.Nu[u]:
                    yj_old = self.yj[j]
                    self.yj[j] += self.lr * (err * self.Nu_minus_half[u] * qi_old - self.reg_param * yj_old)

            print(f"Completed, doing clusters")

            self.u_labels = kmeans_users.fit_predict(self.pu)
            self.i_labels = kmeans_items.fit_predict(self.qi)
            self.y_labels = kmeans_yj.fit_predict(self.yj)

            self.pCu = kmeans_users.cluster_centers_
            self.qCi = kmeans_items.cluster_centers_
            self.yCj = kmeans_yj.cluster_centers_
            
            end = time()
            print(f"Took {round(end - start, 2)} seconds.")
            
        if self.verbose:
            print("Training completed.")

    # Define predict method
    def predict(self, testset):
        cdef list predictions = []
        cdef list actuals = []

        for u, i, actual_rating in testset:
            try:
                inner_user_id = self.trainset.to_inner_uid(u)
                inner_item_id = self.trainset.to_inner_iid(i)
            except ValueError:
                continue
            
            predicted_rating = self._predictor(inner_user_id, inner_item_id)
            predictions.append(predicted_rating)
            actuals.append(actual_rating)

        if not predictions:
            raise ValueError("No valid predictions were made.")

        min_val, max_val = min(predictions), max(predictions)
        new_min, new_max = 1, 5
        scaled_predictions = [(p - min_val) / (max_val - min_val) * (new_max - new_min) + new_min for p in predictions]

        return scaled_predictions, actuals

