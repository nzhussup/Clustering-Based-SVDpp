# Clustering-Based Extended SVD++

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Cython](https://img.shields.io/badge/Cython-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![scikit-surprise](https://img.shields.io/badge/scikit--surprise-FF7043?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## Project Description
This repository contains an implementation of an extended SVD++ model for collaborative filtering, which incorporates clustering techniques to enhance recommendation accuracy. The model builds upon the traditional SVD++ approach by integrating user and item clustering to capture latent group-level preferences. This extension aims to maintain scalability while improving the accuracy of predictions, inspired by advancements in factorized collaborative filtering.

## Background
Traditional factorized collaborative filtering models like SVD++ are highly scalable and form the backbone of many recommendation systems, including those used in Netflix's recommendation engine. This project extends these models by applying a clustering-based approach to both users and items, thereby incorporating shared interests and group behaviors into the factorization process. The concept is based on the research findings from "Clustering-Based Factorized Collaborative Filtering," which demonstrated significant improvements in model performance using such techniques.

## Folder Structure

```plaintext
project_folder/
│
├── src
│   ├── __init__.py
│   ├── build/
│   │   └── ...
│   ├── cluster_rec.py
│   ├── cython_cb_svdpp.c
│   ├── cython_cb_svdpp.cpython-311-darwin.so
│   ├── cython_cb_svdpp.pyx
│   ├── pure_implementation.py
│   └── setup.py
└── test
    ├── custom_cross_validation.py
    ├── cython_imp_test.py
    ├── notebooks/
    │   └── final_notebook.ipynb
    ├── pure_python_imp_test.py
    ├── pypath.txt
    ├── python_cython_comp.py
    ├── surprise-cb-svdpp_test.py
    └── surprise-svdpp_test.py
```

## Models and Versions

During this project, three different models were developed:

- `cluster_rec.py`: This is the CB-SVD++ algorithm built upon the SVD++ class from the surprise library.
- `pure_implementation.py`: This is the implementation of CB-SVD++ from scratch using only NumPy and scikit-learn's KMeans algorithm.
- `cython_cb_svdpp.pyx`: This is the Cythonized version of `pure_implementation.py`. It helped optimize the algorithm and speed up the training process drastically.

Runtime performance on 5 epochs:

- **Cython CB-SVD++**
  - Training: 97.65 seconds
  - Predicting: 0.42 seconds

- **Surprise-based CB-SVD++**
  - Training: 3.66 seconds
  - Predicting: 0.45 seconds

- **Python CB-SVD++**
  - Training: 332.54 seconds
  - Predicting: 10.46 seconds

We recommend using `cython_cb_svdpp.pyx` or `cluster_rec.py` because these models run much faster.

## Installation
To use this extended SVD++ model, you will need Python along with several dependencies related to data handling and matrix operations.

### Prerequisites
- Python 3.x
- NumPy
- Surprise (a Python scikit for building and analyzing recommender systems)

### Setup
**Surprise based model**

Install the required Python packages:
```bash
pip install numpy scikit-surprise
```

**Cython based model**

Install the `cython_cb_svdpp.cpython-311-darwin.so` and put this file into the same folder as your run python script or Jupyter notebook.

### Usage
To run the extended SVD++ model with clustering, you can use the provided scripts. Here's an example of how to execute the model on your dataset:

**Surprise based model**

```python
from cluster_rec import CB_SVDpp

# Load your dataset
data = load_data('your_dataset.csv')

# Initialize the model
model = CB_SVDpp(num_clusters=50, alpha=0.15, n_epochs=50, verbose=True)

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

**Cython based model**

```python
import cython_cb_svdpp

# Load your dataset
data = load_data('your_dataset.csv')

# Initialize the model
model = cython_cb_svdpp.CB_SVDpp(num_clusters=50, alpha=0.15, n_epochs=50, verbose=True)

# Fit the model
model.fit(trainset)

# Make predictions
predictions, actual_ratings = model.predict(testset)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - actual) ** 2))
print("RMSE:", rmse)
```

Replace `'your_dataset.csv'` with your actual data file.

## Acknowledgments
This implementation is inspired by the paper "Clustering-Based Factorized Collaborative Filtering" by Nima Mirbakhsh and Charles X. Ling from Western University. The authors explored the integration of clustering into traditional factorization models, providing a foundation for this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

This revised version addresses minor grammar and formatting issues while maintaining the clarity and completeness of the original content.
