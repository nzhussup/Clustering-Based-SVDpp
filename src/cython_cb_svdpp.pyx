# cython: language_level=3
# Import the necessary components for Cython to handle C-level operations
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
import numpy as np
cimport numpy as cnp
from sklearn.cluster import KMeans
from collections import defaultdict

cnp.import_array()

cdef class CB_SVDpp:
    cdef:
        int num_clusters, n_factors, n_epochs, verbose
        object random_state
        double alpha, lr, reg_param, init_mean, init_std_dev
        double[:, :] pu, pCu, qi, qCi, yj, yCj
        double[:] bu, bi
        int[:] u_labels, i_labels, y_labels
        object Nu_minus_half
        object Nu
        object trainset
        object testset

    def __init__(self, int n_factors=20, int n_epochs=20, double init_mean=0,
                 double init_std_dev=0.1, double reg_param=0.1, double lr=0.01,
                 object random_state=None, int verbose=False, double alpha=0.15, int num_clusters=50):
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.n_factors = n_factors
        self.lr = lr
        self.reg_param = reg_param
        self.pu = np.random.normal(self.init_mean, self.init_std_dev, (1, self.n_factors))
        self.pCu = np.random.normal(self.init_mean, self.init_std_dev, (1, self.n_factors))
        self.qi = np.random.normal(self.init_mean, self.init_std_dev, (1, self.n_factors))
        self.qCi = np.random.normal(self.init_mean, self.init_std_dev, (1, self.n_factors))
        self.yj = np.random.normal(self.init_mean, self.init_std_dev, (1, self.n_factors))
        self.yCj = np.random.normal(self.init_mean, self.init_std_dev, (1, self.n_factors))
        self.u_labels = np.zeros(1, dtype=np.intc)
        self.i_labels = np.zeros(1, dtype=np.intc)
        self.y_labels = np.zeros(1, dtype=np.intc)
        self.bi = np.zeros(1)
        self.bu = np.zeros(1)
        self.Nu_minus_half = defaultdict(float)
        self.Nu = defaultdict(set)
        self.trainset = None
        self.testset = None

    cpdef void calc_Nu(self, trainset):
        cdef int u, i
        for u, i, _ in trainset.all_ratings():
            self.Nu[u].add(i)
        for u in self.Nu:
            self.Nu_minus_half[u] = 1 / sqrt(len(self.Nu[u]))
        self.Nu = {u: list(items) for u, items in self.Nu.items()}

    cdef double _predictor(self, int user_id, int item_id):
        cdef:
            double[:] part1 = np.zeros(self.n_factors, dtype=np.float64)
            double[:] part2 = np.zeros(self.n_factors, dtype=np.float64)
            double[:] sum_result = np.zeros(self.n_factors, dtype=np.float64)
            int j, k
            double result = 0.0

        # Compute part1 and part2, which involve direct contributions from qi, qCi, pu, and pCu
        for k in range(self.n_factors):
            part1[k] = (1 - self.alpha) * self.qi[item_id, k] + self.alpha * self.qCi[self.i_labels[item_id], k]
            part2[k] = (1 - self.alpha) * self.pu[user_id, k] + self.alpha * self.pCu[self.u_labels[user_id], k]

        # Compute the sum_result which aggregates contributions from all items j in Nu(user_id)
        for j in self.Nu[user_id]:
            cluster_id = self.i_labels[j]
            for k in range(self.n_factors):
                sum_result[k] += ((1 - self.alpha) * self.yj[j, k] + self.alpha * self.yCj[cluster_id, k])

        # Apply the Nu_minus_half factor to sum_result and add it to part2
        for k in range(self.n_factors):
            part2[k] += self.Nu_minus_half[user_id] * sum_result[k]

        # Dot product of part1 and part2 for final prediction result
        for k in range(self.n_factors):
            result += part1[k] * part2[k]

        # Add the bias terms for the item and the user
        result += self.bi[item_id] + self.bu[user_id]

        return result


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
        kmeans_users = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        kmeans_items = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        kmeans_yj = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)


        # Fit clusters
        self.u_labels = kmeans_users.fit_predict(self.pu)
        self.i_labels = kmeans_items.fit_predict(self.qi)
        self.y_labels = kmeans_yj.fit_predict(self.yj)

        # Initialize cluster centers
        self.pCu = kmeans_users.cluster_centers_
        self.qCi = kmeans_items.cluster_centers_
        self.yCj = kmeans_yj.cluster_centers_

        # SGD
        cdef double[:] pu_old, qi_old, yj_old
        for epoch in range(self.n_epochs):
            if self.verbose:
                print(f"Processing epoch {epoch}")
            for u, i, r_ui in trainset.all_ratings():
                err = r_ui - self._predictor(u, i)

                # Update biases
                self.bu[u] += self.lr * (err - self.reg_param * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg_param * self.bi[i])

                # Save old values for updates
                pu_old = self.pu[u].copy()
                qi_old = self.qi[i].copy()

                # Update user and item factors
                for j in range(self.n_factors):
                    self.pu[u, j] += self.lr * (err * qi_old[j] - self.reg_param * self.pu[u, j])
                    self.qi[i, j] += self.lr * (err * pu_old[j] - self.reg_param * self.qi[i, j])

                # Update yj and yCj for all items j in Nu(u)
                for j in self.Nu[u]:
                    yj_old = self.yj[j].copy()
                    for k in range(self.n_factors):
                        self.yj[j, k] += self.lr * (err * self.Nu_minus_half[u] * qi_old[k] - self.reg_param * self.yj[j, k])

            # Re-fit clusters with updated factors
            self.u_labels = kmeans_users.fit_predict(self.pu)
            self.i_labels = kmeans_items.fit_predict(self.qi)
            self.y_labels = kmeans_yj.fit_predict(self.yj)

            self.pCu = kmeans_users.cluster_centers_
            self.qCi = kmeans_items.cluster_centers_
            self.yCj = kmeans_yj.cluster_centers_

            if self.verbose:
                print("Epoch completed.")
                print(f"Took {round(t2-t2, 2)} seconds.")

            

        if self.verbose:
            print("Training completed.")


    def predict(self, testset):

        self.testset = testset
        cdef list predictions = []
        cdef list actuals = []
        cdef int inner_user_id, inner_item_id
        cdef double predicted_rating

        # Iterate over each entry in the testset to generate predictions
        for u, i, actual_rating in testset:
            try:
                inner_user_id = self.trainset.to_inner_uid(u)
                inner_item_id = self.trainset.to_inner_iid(i)
                predicted_rating = self._predictor(inner_user_id, inner_item_id)
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except ValueError:
                # Handle cases where the user or item is not in the training set
                continue

        if not predictions:
            raise ValueError("No valid predictions were made.")

        # Scale the predictions
        cdef double min_val = min(predictions)
        cdef double max_val = max(predictions)
        cdef double new_min = 1
        cdef double new_max = 5
        scaled_predictions = [(p - min_val) / (max_val - min_val) * (new_max - new_min) + new_min for p in
                              predictions]

        return scaled_predictions, actuals






