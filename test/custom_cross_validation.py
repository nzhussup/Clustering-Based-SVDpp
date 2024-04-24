import numpy as np
from sklearn.metrics import mean_squared_error
from time import time

def custom_cross_validation(algo_class, train_test: list, alpha_values: list, cluster_values: list, cv: int, surprise: bool, verbose: bool):
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
            
            if verbose:
                model = algo_class(num_clusters=clusters, alpha=alpha, n_epochs=5, verbose=True)
            else:
                model = algo_class(num_clusters=clusters, alpha=alpha, n_epochs=5, verbose=False)
            model.fit(trainset)
            if surprise:
                model.calc_Nu(trainset)
            
            for _ in range(cv):

                if surprise:
                    y_pred, y_true = model.predict_df(testset)
                else:
                    y_pred, y_true = model.predict(testset)
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