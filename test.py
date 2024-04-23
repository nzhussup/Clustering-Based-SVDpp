import numpy as np
from surprise import Dataset
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from cluster_rec import CB_SVDpp
from time import time

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


# Define a custom cross validator
def custom_cross_validation(algo_class, train_test: list, alpha_values: list, cluster_values: list, cv: int):
    """
    Perform cross-validation to find the best alpha and cluster values.
    
    Args:
        algo_class: The algorithm class to use (e.g., CB_SVDpp).
        data: The dataset to use for training and testing.
        alpha_values (list): List of alpha values to try.
        cluster_values (list): List of cluster counts to try.
        cv (int): Number of splits for cross-validation.
        
    Returns:
        dict: Results for each parameter combination.
        tuple: Best alpha and cluster values based on mean RMSE.
    """
    trainset = train_test[0]
    testset = train_test[1]
    
    results = {}
    for alpha in alpha_values:
        for clusters in cluster_values:
            start = time()
            print(f"Testing alpha={alpha}, clusters={clusters}")
            rmses = []
            
            model = algo_class(num_clusters=clusters, alpha=alpha, n_epochs=20)
            model.fit(trainset)
            model.calc_Nu(trainset)
            
            for _ in range(cv):

                y_pred, y_true = model.predict_df(testset)  # Assuming predict_df returns (predictions, _)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                rmses.append(rmse)
            
            mean_rmse = np.mean(rmses)
            results[(alpha, clusters)] = mean_rmse
            print(f"Mean RMSE: {mean_rmse}")
            end = time()
            print(f"Took {round(end-start, 2)} seconds to fit and predict.")
        
    
    # Find the best parameters
    best_params = min(results, key=results.get)
    print(f"Best params (alpha, clusters): {best_params}, RMSE: {results[best_params]}")
    
    return results, best_params

# Initialize different alpha and cluster values
alpha_values = [0.1, 0.15, 0.2]
cluster_values = [50, 100, 150]

# Cross validate alpha and cluster values. Output best params and total runtime
cv_start = time()
results, best_params = custom_cross_validation(CB_SVDpp, [trainset, testset], alpha_values, cluster_values, cv=3)
cv_end = time()
print(f"CV total runtime: {round(cv_end - cv_start, 2)} seconds.")
