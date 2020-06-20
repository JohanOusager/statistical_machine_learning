import tools
import numpy as np
import math
import scipy.linalg as scp
import matplotlib.pyplot as plt
import time
import kNN
import cv2

def pareto_plot(sorted_data, x="", y ="", title="", as_percent=False, save_as=None):

    sum_arr = []
    part_sum = 0
    for data in sorted_data:
        part_sum += data
        sum_arr.append(part_sum)
    sum_arr = np.array(sum_arr)

    plt.figure()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)

    if (as_percent):
        y = np.arange(0, 101, step=5)
        yp = [str(p) + "%" for p in y]

        axis = plt.gca()
        axis.set_ylim([0, 100])
        plt.yticks(y, yp)

    plt.bar(range(len(sorted_data)), sorted_data)
    plt.plot(sum_arr, color='coral')

    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 12)

    plt.savefig(save_as)


#PCA class
class PCA:

    def __init__(self, data):
        #PC = loadings_columns * obs_columns
        self.loadings = []
        self.lambdas = []

        #reshape so each observation is a column
        data_reshape = np.array(data).transpose()
        data = None

        #do SVD
        self.loadings, self.lambdas, _ = scp.svd(data_reshape)

    #transform data to PC space (reduced is an index) i should probably remove reduced from this one
    def to_PC(self, observation, reduced=None):
        if reduced == None:
            reduced = self.loadings.shape[0]

        observation_reshape = observation.transpose()
        observation_PC = self.loadings[:reduced].dot(observation_reshape)
        return observation_PC.transpose()


    #transform data using multiple levels of a reduced PCA model. i should probably rename to to_PC_reduced
    def to_PC_multi(self, observation, reduced=None):
        if reduced == None:
            return self.to_PC(observation)
        else:
            #find the indexes
            var = self.lambdas/np.sum(self.lambdas)*100
            sum_var = [np.sum(var[:p+1]) for p in range(len(var))]
            index_dict = {}
            s = 0
            for model in reduced:
                for PC_index in range(s, len(sum_var)):
                    if sum_var[PC_index] >= model:
                        index_dict[str(model)] = PC_index+1
                        break

            #works until here

            #get the PCS
            PC_data_dict = {}
            for keys in index_dict.keys():
                PC_data_dict[keys] = self.to_PC(observation, index_dict[keys])

            return PC_data_dict

    def from_PC(self, PC_data):
        #PC data should be pxn, e should be pxp , restored should be pxp * pxn = pxn, which is transposed to nxp
        #extend the PCs with zeros if necessary
        if PC_data.shape[1] < self.loadings.shape[0]:
            PC_data = np.concatenate((PC_data, np.zeros((PC_data.shape[0], self.loadings.shape[0] - PC_data.shape[1]))), axis=1)
        return np.transpose(np.dot(np.linalg.inv(self.loadings), np.transpose(PC_data)))

def task_2_1_1():
    print("Start of task 2.1.1")
    #read data
    data = np.load("data/flattened_all.npy")
    print("Read file with shape ", data.shape)
    data_size = data.shape
    data = np.reshape(data, [40000, 325])

    #do PCA on the data
    myPCA = PCA(data[:, 1:])


    #do a pareto chart of explained variance
    variances = np.squeeze(myPCA.lambdas)
    pareto_plot(variances[:30], x="PCS", y="Variance", title="PC variances", save_as="PC_var_absolute.png")
    pareto_plot(np.sqrt(variances[:100]), x="PCS", y="Variance", title="PC variances", save_as="PC_stddev_absolute.png")

    #normalized
    variances = (variances/np.sum(variances))*100
    pareto_plot(variances, x="PCS", y="Explained variance [%]", as_percent=True, title="PC variances [%]", save_as="PC_var_percent.png")


#does a kNN loop to get mean and std dev accruacy. should be moved to kNN or modified to take any ML class
def kNN_iterative_accuracy_test(data_dict, k, iterations, groups=None, verbose=False):

    if groups==None:
        groups=[50, 50]

    accuracies = np.zeros((iterations, len(k)))
    accuracies = np.squeeze(accuracies)
    for i in range (0, iterations, 1):
        for keys in data_dict.keys():
            val, train = tools.rnd_split(data_dict[keys], groups, verbose=False)
            gt = val[:,0]
            mykNN = kNN.kNN(train)
            pred = mykNN.mass_arg_predict(val, k, verbose=False)
            pred = np.squeeze(pred, axis=2)
            accuracy, _, _ = tools.evaluate(pred, gt)

            accuracies[i] = accuracy
    if verbose:
        print("Mean accuracy of ", np.mean(accuracies, axis=0), "% and standard deviation ", np.std(accuracies, axis=0), "%, for k =", k)

    return np.mean(accuracies), np.std(accuracies)


def task_2_1_23():
    print("Start of task 2.1.2/3")
    #read data
    data = np.load("data/flattened_all.npy")
    print("Read file with shape ", data.shape)
    data_size = data.shape
    data = np.reshape(data, [40000, 325])

    #do PCA on the data
    myPCA = PCA(data[:,1:])

    #get 80%, 90%, 95%, 99% var PCs
    var = myPCA.lambdas/np.sum(myPCA.lambdas)*100
    sum_var = [np.sum(var[:p+1]) for p in range(len(var))]
    index_dict = {"80" : None, "90" : None, "95" : None, "99" : None}
    for p in range(len(sum_var)):
        if (index_dict["80"] == None and sum_var[p] >= 80):
            index_dict["80"] = p+1
        elif (index_dict["90"] == None and sum_var[p] >= 90):
            index_dict["90"] = p+1
        elif (index_dict["95"] == None and sum_var[p] >= 95):
            index_dict["95"] = p+1
        elif (index_dict["99"] == None and sum_var[p] >= 99):
            index_dict["99"] = p+1

    print("Found splits:" , index_dict.items())

    #transform data to PC space
    PC_data = {}
    for keys in index_dict.keys():
        tmp = np.empty([data.shape[0], index_dict[keys]+1])
        tmp[:, 0] = data[:, 0]
        tmp[:, 1:] = myPCA.to_PC(data[:, 1:], index_dict[keys])
        PC_data[keys] = tmp

    k = range(1, 11)
    for keys in PC_data.keys():
        print("kNN for ", keys, "% variance explained")
        val, train = tools.rnd_split(PC_data[keys], [50, 50], verbose=False)
        gt = val[:,0]
        mykNN = kNN.kNN(train)
        pred = mykNN.mass_arg_predict(val, k, verbose=False)
        print(pred.shape, gt.shape)
        pred = np.squeeze(pred, axis=2)
        accuracy, _, _ = tools.evaluate(pred, gt)
        print(accuracy)

        kNN.k_plot((keys + "% PC based kNN "), k, accuracy*100, yticks=(range(93, 101, 1)), ylabel="Accuracy [%]")

    #do kNN 1-10 for 80%, 90%, 95% and 99% variance explained


    #measure run times and compare


def task_2_3_none(data):
    print("using raw data:")

    #do PCA on the data
    myPCA = PCA(data[:,1:])

    #get 80%, 90%, 95%, 99% var PCs
    tmp = myPCA.to_PC_multi(data[:,1:], reduced=[80])

    PC_data = {}
    for keys in tmp.keys():
        labels = np.expand_dims(data[:,0], axis=1)
        PC_data[keys] = np.concatenate((labels, tmp[keys]), axis=1)

    return kNN_iterative_accuracy_test(PC_data, [5], 10, groups=[50, 50], verbose=True)


def task_2_2_nb(data):
    print("using normalized data:")

    #do normalization
    data[:,1:] = tools.normalize(data[:,1:])                                      #the magic

    #do PCA on the data
    myPCA = PCA(data[:,1:])

    #get 80%, 90%, 95%, 99% var PCs
    tmp = myPCA.to_PC_multi(data[:,1:], reduced=[80])

    PC_data = {}
    for keys in tmp.keys():
        labels = np.expand_dims(data[:,0], axis=1)
        PC_data[keys] = np.concatenate((labels, tmp[keys]), axis=1)

    return kNN_iterative_accuracy_test(PC_data, [5], 10, groups=[50, 50], verbose=True)


def task_2_2_na(data):
    print("using normalized PC:")

    #do PCA on the data
    myPCA = PCA(data[:,1:])

    #get 80%, 90%, 95%, 99% var PCs
    tmp = myPCA.to_PC_multi(data[:,1:], reduced=[80])

    PC_data = {}
    for keys in tmp.keys():
        labels = np.expand_dims(data[:,0], axis=1)
        PC_data[keys] = np.concatenate((labels, tools.normalize(tmp[keys])), axis=1)

    return kNN_iterative_accuracy_test(PC_data, [5], 10, groups=[50, 50], verbose=True)


def task_2_2_sb(data):
    print("using standardized data:")

    #do normalization
    data[:,1:] = tools.standardize(data[:,1:])

    #do PCA on the data
    myPCA = PCA(data[:,1:])

    #get 80%, 90%, 95%, 99% var PCs
    tmp = myPCA.to_PC_multi(data[:,1:], reduced=[80])

    PC_data = {}
    for keys in tmp.keys():
        labels = np.expand_dims(data[:,0], axis=1)
        PC_data[keys] = np.concatenate((labels, tmp[keys]), axis=1)

    return kNN_iterative_accuracy_test(PC_data, [5], 10, groups=[50, 50], verbose=True)


def task_2_2_sa(data):
    print("using standardized PC")

    #do PCA on the data
    myPCA = PCA(data[:,1:])

    #get 80%, 90%, 95%, 99% var PCs
    tmp = myPCA.to_PC_multi(data[:,1:], reduced=[80])

    PC_data = {}
    for keys in tmp.keys():
        labels = np.expand_dims(data[:,0], axis=1)
        PC_data[keys] = np.concatenate((labels, tools.standardize(tmp[keys])), axis=1)

    return kNN_iterative_accuracy_test(PC_data, [5], 10, groups=[50, 50], verbose=True)


def task_2_2():
    print("Start of task 2.2")

    #read data
    data = np.load("data/flattened_all.npy")
    print("Read file with shape ", data.shape)
    data_size = data.shape
    data = np.reshape(data, [40000, 325])

    accuracies = np.zeros(5)
    stddev = np.zeros(5)
    accuracies[0], stddev[0] = task_2_3_none(data.copy())
    accuracies[1], stddev[1] = task_2_2_nb(data.copy())
    accuracies[2], stddev[2] = task_2_2_na(data.copy())
    accuracies[3], stddev[3] = task_2_2_sb(data.copy())
    accuracies[4], stddev[4] = task_2_2_sa(data.copy())

    plt.figure()
    plt.bar(x=[1, 2, 3, 4, 5], height=accuracies*100, yerr=stddev*100, tick_label=["Raw", "Normalized data", "Normalized PCs", "Standardized data", "Standardized PCs"])
    plt.tick_params(labelsize=20)
    plt.ylim(bottom=92.5, top=100)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(20, 12)
    plt.title("Accuracy with data preparation (80% PC)", fontsize=20)
    plt.ylabel("Accuracy [%]", fontsize=20)
    plt.savefig("normalOrStandard_effect.png")
    #plt.show()


def task_2_3():
    print("Start of task 2.3")

   #read data
    data = np.load("data/flattened_all.npy")
    print("Read file with shape ", data.shape)
    data_size = data.shape
    data = np.reshape(data, [40000, 325])

    accuracies_smoothing = []
    stddev_smoothing = []
    acc, stddev = task_2_3_none(data.copy())
    accuracies_smoothing.append(acc)
    stddev_smoothing.append(stddev)

    #get images
    images = np.reshape(data[:,1:], (data_size[0], 18, 18))
    images[:] = images.swapaxes(1, 2)

    x = [0]
    sigmas  = range(5, 151, 5)
    for sigma in sigmas:
        print("Sigma = ", sigma*0.01)
        tmp = cv2.GaussianBlur(images[:], ksize=(5,5), sigmaX=sigma*0.01)
        tmp_data = data.copy()
        tmp_data[:,1:] = np.reshape(tmp, (data_size[0], data_size[1]-1))
        acc, stddev = task_2_3_none(tmp_data.copy())
        x.append(sigma*0.01)
        accuracies_smoothing.append(acc)
        stddev_smoothing.append(stddev)

    x = np.array(x)
    accuracies_smoothing = np.array(accuracies_smoothing)*100
    stddev_smoothing = np.array(stddev_smoothing)*100


    plt.figure()
    plt.errorbar(x, accuracies_smoothing, stddev_smoothing, fmt='--')#, marker='s', mfc='red', mec='green', ms=20, mew=4)
    plt.tick_params(labelsize=14)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 12)
    plt.title("Accuracy with smoothing (80% PC)", fontsize=20)
    plt.xlabel("Sigma", fontsize=20)
    plt.ylabel("Accuracy [%]", fontsize=20)
    plt.savefig("smoothing_effect.png")
    #plt.show()

def task_2_4():
    print("Start of task 2.4")

    #read data
    data = np.load("data/flattened_all.npy")
    print("Read file with shape ", data.shape)
    data_size = data.shape
    data = np.reshape(data, [40000, 325])


#plot one of each cipher
    print("Drawing originals")
    #get images
    images = np.reshape(data[:,1:], (data_size[0], 18, 18))
    images[:] = images.swapaxes(1, 2)

    c = 0
    for i in range(50, 4000, 400):
        #cv2.imshow("cipher", images[i])
        #cv2.waitKey()
        cv2.imwrite("ciphers/cipher_" + str(c) + "_orig.png",  cv2.resize(images[i]*255, (144,144), interpolation=cv2.INTER_NEAREST))
        c += 1


#make them into PCA and back again, to show the lost information
    myPCA = PCA(data[:,1:])

    PC_data_dict = myPCA.to_PC_multi(data[:,1:], reduced=[80, 90, 95])
    for keys in PC_data_dict:
        print("Drawing PC" + keys)
        restored_data = myPCA.from_PC(PC_data_dict[keys])
        print(np.sum(restored_data - data[:,1:]), np.min(restored_data))
        images = np.reshape(restored_data, (data_size[0], 18, 18))
        c = 0
        for i in range(50, 4000, 400):
            #cv2.imshow("cipher", images[i])
            #cv2.waitKey()
            cv2.imwrite("ciphers/cipher_" + str(c) + "_" + str(keys) + "%_restored.jpg", cv2.resize(images[i]*255, (144,144), interpolation=cv2.INTER_NEAREST))
            c += 1

#take 10 eigenvectors
#transform them to data
    #make that data into images

    #reconstruct the same 10 images using 80, 90, 95
    #show the images

    #compare two different ciphers in PC space
    #compare the mean of those two ciphers in PC space
    #compare to loadings



###main
#task_2_1_1()
#task_2_1_23()
#task_2_2()
#task_2_3()

#doesnt work
#task_2_4()
