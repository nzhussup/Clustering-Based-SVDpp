from surprise import Dataset, SVDpp, accuracy
from surprise.model_selection import train_test_split
from time import time

# Import and split data
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)

# Train and test on random values
algo = SVDpp(n_epochs=20, verbose=True)
start = time()
algo.fit(trainset)
preds = algo.test(testset)
accuracy.rmse(preds)
end = time()
print(f"Took {round(end - start, 2)} seconds to fit and predict.")
