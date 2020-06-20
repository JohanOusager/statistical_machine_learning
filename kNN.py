import random
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import time


#Takes a flattened list of matrices from R and restores the dimensions
#this is neccesary because when saving the list are collapsed in C style (all of list 1, then all of list 2)
#while the matrix is collapsed in Fortran style (first element of row 1, then first element of row 2)
def restore3D_Rda(flattened_R_data, orig_dims):
    restored = np.empty(orig_dims, dtype=np.double)
    for i in range(orig_dims[0]):
        slice = flattened_R_data[ i*orig_dims[1]*orig_dims[2] : (i+1)*orig_dims[1]*orig_dims[2]  ]
        restored_slice = np.reshape(slice, [orig_dims[1], orig_dims[2]], order='F')
        restored[i] = restored_slice
    return restored


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

#make and save plot
def k_plot(title, k, y, yticks=(0, 100, 5), ylabel="Accuracy"):
    plt.figure()
    plt.plot(k, y, '--bo')
    plt.ylabel(ylabel, fontsize=20)
    plt.yticks(yticks)
    plt.xlabel('k', fontsize=20)
    plt.xticks(range(1, np.max(k)+1, 1))
    plt.title(title + "set " + ylabel, fontsize=20)
    plt.tick_params(labelsize=14)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 12)
    plt.savefig(title + '_knn.png')


#kNN class
class kNN:


    #"train" by saving the data for later comparison
    def __init__(self, data):
        self.data = data


    #predict the classes of multiple observations
    def mass_predict(self, pred_data, k, verbose=False):
        predictions = np.array([self.predict(obs, k, verbose) for obs in pred_data])
        if verbose:
            print("Made ", predictions.shape[0] ,"predictions")
        return predictions


    #predict the class of an observation
    def predict(self, obs, k, verbose=False):
        nearest_neighbours = sorted(list(self.data), key=lambda neighbour: np.linalg.norm(obs[1:] - neighbour[1:]))
        k_max = np.max(k)
        nearest_k = np.array(nearest_neighbours[0:k_max])
        classes = np.array(nearest_k[:, 0], np.int64)
        mode = np.array([stats.mode(classes[:n]) for n in k])
        if verbose:
            #print("Neighbour classes were", classes)
            print("predicted:", mode[:,0])
        return mode[:,0]


    #predict the classes of multiple observations
    def mass_arg_predict(self, pred_data, k, verbose=False):
        predictions = np.array([self.arg_predict(obs, k, verbose) for obs in pred_data])
        if verbose:
            print("Made ", predictions.shape[0] ,"predictions")
        return predictions


    #argpartition prediction
    def arg_predict(self, obs, k, verbose=False):
        #tttttttt = time.time()
        #old_t = time.time()

        #get distance array
        #t = time.time()
        diff = self.data[:,1:] - obs[1:]#.reshape(-1, 1)
        #tt = time.time()-t
        #print("subtracting", tt)

        #ttt = time.time()
        diff = np.linalg.norm(diff, axis=1)
        #tttt = time.time()-ttt
        #print("norm", tttt)

        #new_t = time.time()
        #print("calc diff", new_t-old_t)
        #old_t = time.time()

        #argpartition
        partition = np.argpartition(diff[:], k)

        #new_t = time.time()
        #print("partition", new_t-old_t)
        #old_t = time.time()

        #get labels
        labels = np.squeeze(self.data[:,:1])

        #get labels of the nearest neighbours
        classes = np.array([np.take(labels, partition[:_k]) for _k in k])

        mode = np.array([stats.mode(classes[n]) for n in range(classes.shape[0])], dtype=np.int64)

        #new_t = time.time()
        #print("finding mode", new_t-old_t)


        if verbose:
            #print("Neighbour classes were", np.array(classes))
            print("predicted:", mode[:,0])

        #print("Total", time.time()-tttttttt)
        return mode[:,0]


##main##

###1.4.1/1.4.2 - 50/50 split at varying k
def task_1_4_12():
    #read data
    data = np.load("data/id100.npy")
    print("Read file with shape ", data.shape)

    #split data
    val, train = rnd_split(data, [50, 50], verbose=True)
    #train kNN
    mykNN = kNN(train)
    k = range(1, 51)

    print("Training set")
    train_start = time.time()
    #predictions = mykNN.mass_predict(train[:], k, verbose=False)
    predictions = mykNN.mass_arg_predict(train[:], k, verbose=False)
    train_end = time.time()
    train_time = train_end - train_start
    predictions = np.squeeze(predictions, axis=2)
    ground_truth = np.int64(train[:, 0])

    print(ground_truth.shape, predictions.shape)

    accuracy, _, _ = evaluate(predictions, ground_truth, verbose=True)
    print(accuracy[0]*100)

    k_plot("Individual training ", k, accuracy*100)

    print("Validation set")
    val_start = time.time()
    #predictions = mykNN.mass_predict(val[:], k, verbose=False)
    predictions = mykNN.mass_arg_predict(val[:], k, verbose=False)
    val_end = time.time()
    val_time = val_end - val_start
    predictions = np.squeeze(predictions)
    ground_truth = np.int64(val[:, 0])

    print(ground_truth.shape, predictions.shape)

    accuracy, _, _ = evaluate(predictions, ground_truth, verbose=True)


    k_plot("Individual validation ", k, accuracy*100)

    print("Predictions for task 1.4.1 took ", int(train_time), "(training) and", int(val_time), "(validation) seconds")
    #plt.show()




###1.4.3 90/10 split with 10 repetitions
def task_1_4_3():
    accuracies = []
    for i in range(10):
        #read data
        data = np.load("data/id100.npy")
        #split data
        val, train = rnd_split(data, [10, 90], verbose=False)
        #train kNN
        mykNN = kNN(train)
        k = [5]

        #predictions = mykNN.mass_predict(val[:], k, verbose=False)
        predictions = mykNN.mass_arg_predict(val[:], k, verbose=False)
        predictions = np.squeeze(predictions, axis=2)
        ground_truth = np.int64(val[:, 0])
        accuracy, _, _ = evaluate(predictions, ground_truth, verbose=False)
        accuracies.append(accuracy)
    print("mean accuracy ", np.mean(accuracies))
    print("std deviation", np.std(accuracies))


###1.4.4 person independant kNN
def task_1_4_4_all_persons():
    print("1_4_4_all_persons")
    #read data
    data = np.load("data/flattened_all.npy")
    print("Read file with shape ", data.shape)
    data = np.reshape(data, [40000, 325])

    val, train = rnd_split(data, [50, 50], verbose=False)
    val = np.array(val)
    train = np.array(train)

    #train kNN
    mykNN = kNN(train)
    k = range(1, 51)

    """
    print("Training set")
    train_start = time.time()
    #predictions = mykNN.mass_predict(train[:], k, verbose=False)
    predictions = mykNN.mass_arg_predict(train[:], k, verbose=False)
    train_end = time.time()
    train_time = train_end - train_start

    predictions = np.squeeze(predictions)
    ground_truth = np.int64(train[:, 0])
    accuracy, _, _ = evaluate(predictions, ground_truth, verbose=False)

    k_plot("All persons training ", k, accuracy*100)
    
    """

    print("Validation set")
    val_start = time.time()
    #predictions = mykNN.mass_predict(val[:], k, verbose=False)
    predictions = mykNN.mass_arg_predict(val[:], k, verbose=False)
    val_end = time.time()
    val_time = val_end - val_start
    predictions = np.squeeze(predictions, axis=2)
    ground_truth = np.int64(val[:, 0])
    accuracy, _, _ = evaluate(predictions, ground_truth, verbose=True)

    k_plot("All persons validation ", k, accuracy*100, ylabel="Accuracy [%]")

    print("Predictions for task 1.4.4 took ", int(val_time), "(validation) seconds")


###1.4.5 disjust
def task_1_4_4_disjunct():
    print("1_4_4_disjunct")
    #read data
    data = np.load("data/flattened_all.npy")
    print("Read file with shape ", data.shape)
    data = np.reshape(data, [40000, 325])

    val   = data[:int(data.shape[0]/2) , :]
    train = data[int(data.shape[0]/2): , :]
    val = np.array(val)
    train = np.array(train)

    #train kNN
    mykNN = kNN(train)
    k = range(1, 51)

    """
    print("Training set")
    predictions = mykNN.mass_predict(train[:], k, verbose=False)
    predictions = np.squeeze(predictions)
    ground_truth = np.int64(train[:, 0])
    accuracy, _, _ = evaluate(predictions, ground_truth, verbose=False)
    
    k_plot("Disjunct persons validation ", k, accuracy*100)
    """

    print("Validation set")
    #predictions = mykNN.mass_predict(val[:], k, verbose=False)
    predictions = mykNN.mass_arg_predict(val[:], k, verbose=False)
    predictions = np.squeeze(predictions, axis=2)
    ground_truth = np.int64(val[:, 0])
    accuracy, _, _ = evaluate(predictions, ground_truth, verbose=True)

    k_plot("Disjunct validation ", k, accuracy*100, ylabel="Accuracy [%]")


###########################


#set seed
np.random.seed(423)
#run tasks
#task_1_4_12()
#task_1_4_3()
#task_1_4_4_all_persons()
#task_1_4_4_disjunct()
