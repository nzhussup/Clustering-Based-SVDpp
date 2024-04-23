import numpy as np
import pandas as pd
from surprise import SVDpp
from surprise import Dataset
from sklearn.cluster import KMeans
from collections import defaultdict
from functools import lru_cache


class CB_SVDpp:
    
    def __init__(self, num_clusters:int, alpha:float, n_epochs:int, random_state = 123, verbose = False):
        
        # Init params
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.verbose = verbose
        
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
        
        ## Algorithm
        self.algo = None
        
        
        
    def fit(self, trainset):
        """
        Fit's the CB-SVD++ algorithm.

        Args:
            trainset (_surprise dataset_): _description_
        """
        
        algo = SVDpp(random_state = self.random_state, n_epochs=self.n_epochs, verbose=self.verbose)
        algo.fit(trainset)
        
        self.algo = algo

        self.pu = algo.pu
        self.qi = algo.qi     
        self.yj = algo.yj
        self.bi = algo.bi
        self.bu = algo.bu
        
        # Clustering user features to find pCu
        kmeans_user = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        self.u_labels = kmeans_user.fit_predict(self.pu)
        self.pCu = kmeans_user.cluster_centers_

        # Clustering item features to find qCi
        kmeans_item = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        self.i_labels = kmeans_item.fit_predict(self.qi)
        self.qCi = kmeans_item.cluster_centers_

        # Clustering yj features to find yCj
        kmeans_yj = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        self.y_labels = kmeans_yj.fit_predict(self.yj)
        self.yCj = kmeans_yj.cluster_centers_
        
        print("Model fitted! Parameters updated.")
        
        
        
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
                
    @lru_cache(maxsize=None) 
    def _predictor_function(self, user_id, item_id):
        
        """
        Modified predictor function

        Returns:
            _type_: _description_
        """
        
        user_id = int(user_id)
        item_id = int(item_id)
        
        part1 = ((1-self.alpha)*self.qi[item_id] + self.alpha*self.qCi[self.i_labels[item_id]]).T
        part2 = ((1-self.alpha)*self.pu[user_id] + self.alpha*self.pCu[self.u_labels[user_id]])

        sum_result = np.zeros_like(self.pu[user_id])   # Use 'pu' dimension as a reference for initializing 'sum_result'
        
        for j in self.Nu[user_id]:  # Iterating over items in N(u)
            cluster_id = self.i_labels[j]
            term = (1 - self.alpha) * self.yj[j] + self.alpha * self.yCj[cluster_id]  # Accessing yCj with cluster_id
            sum_result += term

        intermediate = part2 + (self.Nu_minus_half[user_id] * sum_result)
        res = np.dot(part1, intermediate) + self.bi[item_id] + self.bu[user_id] # Adding biases at the end
        return res
    
    def predict_user(self, testset, user_id: int, item_id: int):
        """_summary_

        Args:
            testset (_surprise df_): _description_
            user_id (int): _description_
            item_id (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _returns scaled prediction and actual rating for user on item_
        
        Example usage:    
            y_pred, y_true = algo.predict_user(user_id, item_id)
        """
        
        
        try:
            user_id, item_id = self.algo.trainset.to_inner_uid(str(user_id)), self.algo.trainset.to_inner_iid(str(item_id))
            predictions, y_true = [], []

            
            
            user_pred = self.predictor_function(user_id, item_id)
            
            
            for u, i, actual_rating in testset:
                
                try:
                    predicted_rating = self._predictor_function(self.algo.trainset.to_inner_uid(str(u)), 
                                                               self.algo.trainset.to_inner_iid(str(i)))

                    predictions.append(predicted_rating)
                    y_true.append(actual_rating)
                except Exception as e:
                    pass
            
            min_val, max_val = min(predictions), max(predictions)
            new_min, new_max = 1, 5
            
            scaled_user_pred = (user_pred - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

            # Initialize the rating as None to indicate if the rating wasn't found
            rating = None

            # Search for the rating in the testset
            for uid, iid, r in testset:
                if uid == user_id and iid == item_id:
                    rating = r
                    break
                
            
            return scaled_user_pred, rating
        except Exception as e:
            print("Error! User unknown!")
            raise ValueError
        
    def predict_df(self, testset):
        """Generates and scales predictions for all user-item pairs in the testset,
        returning both the scaled predictions and actual ratings.

        Args:
            testset (list): The test dataset containing tuples of (user, item, actual rating).

        Returns:
            tuple: A tuple containing two lists - scaled predictions and actual ratings.
        """
        predictions, actuals = [], []

        # Iterate over each entry in the testset to generate predictions
        for u, i, actual_rating in testset:
            try:
                # Convert raw user and item IDs to inner IDs used by the Surprise training set
                inner_user_id = self.algo.trainset.to_inner_uid(str(u))
                inner_item_id = self.algo.trainset.to_inner_iid(str(i))
                
                # Use the predictor function to estimate the rating
                predicted_rating = self._predictor_function(inner_user_id, inner_item_id)
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except ValueError:
                # This exception handles cases where the user or item is not in the training set
                continue

        if not predictions:
            raise ValueError("No valid predictions were made.")

        # Scale the predictions
        min_val, max_val = min(predictions), max(predictions)
        new_min, new_max = 1, 5  # Define the scale range
        scaled_predictions = [(p - min_val) / (max_val - min_val) * (new_max - new_min) + new_min for p in predictions]

        return scaled_predictions, actuals




        
                
        
        
        
        
        
    
        