import numpy as np
from surprise import AlgoBase, Dataset
from surprise.model_selection import train_test_split
from time import time
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from collections import defaultdict

class CB_SVDpp:


    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std_dev=.1,
                reg_param = .1, lr=.01,
                 random_state=None, verbose = False, 
                 alpha = 0.15, num_clusters = 50
                 ):
        
        # Init params
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.n_factors = n_factors
        self.lr =lr
        self.reg_param = reg_param
        
        
        # Init features
        ## User feat / User cluster feat
        self.pu = None
        self.pCu = None
        self.u_labels = None
        
        ## Item feat / Item cluster feat
        self.qi = None
        self.qCi = None
        self.i_labels = None
        
        ## Y feat / Y cluster feat
        self.yj = None
        self.yCj = None
        self.y_labels = None
        
        ## Biases
        self.bi = None
        self.bu = None
        
        ## NU and NU minus half
        self.Nu_minus_half = defaultdict(float)
        self.Nu = defaultdict(set)
        
        
    def calc_Nu(self, trainset):
        
        """
        Calculates the implicit feedbacks of the users
        Use before training
        """
        for u, i, _ in trainset.all_ratings():
            self.Nu[u].add(i)

        # Calculate |N(u)|^-0.5 for each user
        for u in self.Nu:
            self.Nu_minus_half[u] = 1 / np.sqrt(len(self.Nu[u]))

        # Convert Nu from a set of item indices to a list for easier processing later
        self.Nu = {u: list(items) for u, items in self.Nu.items()}
                

    def _predictor(self, user_id, item_id):
        
        user_id = int(user_id)
        item_id = int(item_id)
        
        part1 = ((1-self.alpha)*self.qi[item_id] + self.alpha*self.qCi[self.i_labels[item_id]]).T
        part2 = ((1-self.alpha)*self.pu[user_id] + self.alpha*self.pCu[self.u_labels[user_id]])

        sum_result = np.zeros_like(self.pu[user_id]) 
        
        for j in self.Nu[user_id]:
            cluster_id = self.i_labels[j]
            term = (1 - self.alpha) * self.yj[j] + self.alpha * self.yCj[cluster_id]
            sum_result += term

        intermediate = part2 + (self.Nu_minus_half[user_id] * sum_result)
        res = np.dot(part1, intermediate) + self.bi[item_id] + self.bu[user_id] # Adding biases at the end
        return res
    
        
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
            
    def predict(self, testset):
        predictions = []
        actuals = []

        # Iterate over each entry in the testset to generate predictions
        for u, i, actual_rating in testset:
            # Convert raw user and item IDs to inner IDs used by the Surprise training set
            try:
                inner_user_id = self.trainset.to_inner_uid(u)
                inner_item_id = self.trainset.to_inner_iid(i)
            except ValueError:
                # This exception handles cases where the user or item is not in the training set
                continue
            
            # Use the predictor function to estimate the rating
            predicted_rating = self._predictor(inner_user_id, inner_item_id)
            predictions.append(predicted_rating)
            actuals.append(actual_rating)

        if not predictions:
            raise ValueError("No valid predictions were made.")

        # Optionally scale the predictions
        min_val, max_val = min(predictions), max(predictions)
        new_min, new_max = 1, 5  # Define the scale range
        scaled_predictions = [(p - min_val) / (max_val - min_val) * (new_max - new_min) + new_min for p in predictions]

        return scaled_predictions, actuals

        
        
# algo = CB_SVDpp(n_epochs=20, verbose=True) # TAKES TOO LONG!!!
# data = Dataset.load_builtin('ml-100k')
# trainset, testset = train_test_split(data, test_size=0.2)

# algo.fit(trainset)
# y_pred, y_true = algo.predict(testset)
# print(np.sqrt(mean_squared_error(y_true, y_pred)))

        

