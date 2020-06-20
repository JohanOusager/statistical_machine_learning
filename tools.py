import numpy as np

#randomly split data into group_percentiles.size groups
def rnd_split(arr, group_percentiles, verbose=False):
    data_size = arr.shape[0]
    #make array of random nummbers
    RNJesus_array = np.random.rand(data_size)

    #make mask
    mask = np.zeros(data_size, dtype=np.int64)

    sum_percentile = 0
    c = 0
    for groups in group_percentiles:
        c += 1
        sum_percentile += groups
        delimiter = np.percentile(RNJesus_array, sum_percentile)
        mask[RNJesus_array > delimiter] = c

    out = []
    for groups in range(c):
        new_group = arr[mask == groups]
        out.append(new_group)
        if verbose:
            print("Created group ", groups, " with ", new_group.shape[0], " elements.")
    return out


#compare predictions to ground truth
def evaluate(prediction, ground_truth, verbose=False):
    correct = (prediction == ground_truth.reshape(-1, 1))
    sums = np.sum(correct, axis=0)
    accuracy = sums / len(ground_truth)
    if verbose:
        print("corrects: ", sums, "/", ground_truth.size)
        print("accuracy ", accuracy)
    return accuracy, sums, correct


#array = (array - array min) / (array max - array min)
def normalize(data):
    centered = data - np.min(data, axis=0)
    normalized = centered / np.max(centered, axis=0)
    return normalized


#array = ( array - arraymean ) / array std dev
def standardize(data):
    centered = data - np.mean(data, axis=0)
    standardized = centered / np.std(centered, axis=0)
    return standardized

def tensorflow_format(data, classes=10):
    labels = np.zeros([data.shape[0], 10])
    labels[np.arange(data.shape[0]), data[:,0].astype(int)] = 1
    return  data[:, 1:], labels