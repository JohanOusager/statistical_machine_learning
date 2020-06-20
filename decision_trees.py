import numpy as np
import scipy
import matplotlib.pyplot as plt
import PCA

#split data into one or more groups
def split(data, decision_points, index=1):
    groups = []
    decision_points = decision_points

    #for binary split
    if np.isscalar(decision_points):
        column = data[:, index]
        where = np.argwhere(column < decision_points)
        groups.append(np.squeeze(data[where, :]))
        data = np.delete(data, where, axis=0)
        groups.append(np.squeeze(data))

    #for multiple group split
    else:
        for dp in decision_points:
            column = data[:, index]
            where = np.argwhere(column < dp)
            groups.append((np.squeeze(data[where, :])))
            data = np.delete(data, where, axis=0)
        #add the last boi
        groups.append(np.squeeze(data))

    return np.squeeze(np.array(groups))


#binarize data
def binarize(data, decision_point):
    return np.greater(data, decision_point)


#get the entropy of a sample
def get_entropy(labeled_data):
    counts = np.bincount(np.array(labeled_data[:,0], dtype=int))
    entropy = 0
    for count in counts:
        if count > 0:
            p = count / labeled_data.shape[0]
            entropy += -p * np.log2(p)
    return entropy


#get the entropy of a sample after splitting
def get_entropy_groups(groups):
    #entropy after split
    all_size = np.sum(np.array([groups[i].shape for i in range(groups.shape[0])]))
    entropy = 0
    for group in groups:
        entropy += ( group.shape[0] * get_entropy(group) ) / all_size
    return entropy


#task 4.1 - information gain using different decision points
def task_4_1():
    #load a boi
    data = np.load("data/new_100_corner.npy")
    data = data[18]
    print(data.shape)

    #do PCA to get top 5 PCs
    myPCA = PCA.PCA(data[:, 1:])
    data[:, 1:] = myPCA.to_PC(data[:, 1:])
    data = data[:, :6]
    print(np.min(data, axis=0))
    print(np.max(data, axis=0))


    #make binary data dict
    decision_points = np.arrange(-10.0, 10.0, 0.1)
    PC_range = range(0, 5, 1)
    bin_dict = {}
    for PC in PC_range:
        bin_data = {}
        for dp in decision_points:
            bin_data[dp] = split(data, dp, index=PC)
        bin_dict[PC] =  bin_data

    #calc start entropy
    raw_entropy = get_entropy(data)
    print("Start entropy:", raw_entropy)

    #calculate entropy for each PC at each dp
    entropies = []
    for PC in PC_range:
        entropy = []
        for dp in decision_points:
            entropy.append(raw_entropy - get_entropy_groups(bin_dict[PC][dp]))
        entropies.append(entropy)

    #plot that garbage
    plt.figure()
    plt.xlabel("Decision point")
    plt.ylabel("Information gain")

    for entropy in entropies:
        plt.plot(decision_points, entropy)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 12)

    plt.show()
    #plt.savefig(save_as)


###main
task_4_1()

