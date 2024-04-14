# Clustering-Based Extended SVD++

## Project Description
This repository contains an implementation of an extended SVD++ model for collaborative filtering, incorporating clustering techniques to enhance recommendation accuracy. The model builds upon the traditional SVD++ approach by integrating user and item clustering to capture latent group-level preferences. This extension aims to maintain scalability while improving the accuracy of predictions, inspired by advancements in factorized collaborative filtering.

## Background
Traditional factorized collaborative filtering models like SVD++ are highly scalable and form the backbone of many recommendation systems, including those used in Netflix's recommendation engine. This project extends these models by applying a clustering-based approach to both users and items, thereby incorporating shared interests and group behaviors into the factorization process. The concept is based on the research findings from "Clustering-Based Factorized Collaborative Filtering," which showed significant improvements in model performance using such techniques.

## Installation
To use this extended SVD++ model, you will need Python along with several dependencies related to data handling and matrix operations.

### Prerequisites
- Python 3.x
- NumPy
- Surprise (a Python scikit for building and analyzing recommender systems)

### Setup

Install the required Python packages:
```bash
pip install numpy scikit-surprise
```

## Usage
To run the extended SVD++ model with clustering, you can use the provided scripts. Here's an example of how to execute the model on your dataset:

```python
from clustering_svdpp import ClusteringSVDpp

# Load your dataset
data = load_data('your_dataset.csv')

## THE MODEL ACCEPTS ONLY SURPRISE DATASETS
## SO CONVERT YOUR DATA INTO SURPRISE FORMAT

# Initialize the model
model = ClusteringSVDpp(num_clusters = 50, alpha = 0.15, n_epochs = 50, verbose True)

# Fit the model
model.fit(trainset)

# Calculate implicit feedbacks
model.calc_Nu(trainset)

# Make predictions
predictions, actual_ratings = model.predict_df(testset)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - actual) ** 2))
print("RMSE:", rmse)
```

Replace `'your_dataset.csv'` with your actual data file.


## Acknowledgments
This implementation is inspired by the paper "Clustering-Based Factorized Collaborative Filtering" by Nima Mirbakhsh and Charles X. Ling from Western University. The authors explored the integration of clustering into traditional factorization models, providing a foundation for this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
