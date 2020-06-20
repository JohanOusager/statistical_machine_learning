import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import tools
import tensorflow
from tensorflow import keras
import pickle

class ANN:

    def __init__(self, layers, activations, optimizer=keras.optimizers.SGD(), loss=keras.losses.CategoricalCrossentropy()):
        layer_arr = []
        for layer, activation in zip(layers, activations):
            layer_arr.append(keras.layers.Dense(layer, activation=activation))
        self.model = keras.Sequential(layer_arr)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    def train(self, train, epochs, batch_size=32, validation_freq=1, val=None):
        #split data
        if val is None:
            train, val = tools.rnd_split(data, [50, 50])


        #make data the right format
        train_d, train_l = tools.tensorflow_format(train)
        val_d, val_l = tools.tensorflow_format(val)

        self.model.fit(train_d, train_l, epochs=epochs, batch_size=batch_size, validation_data=(val_d, val_l), validation_freq=validation_freq)


def accuracy_plot(accuracies, label=None, figure=None, title=None):
    if title is None:
        title = "ANN accuracy"
    if figure is None:
        figure = plt.figure(1, figsize=(16, 10))
    epochs = range(len(accuracies))
    plt.plot(epochs, accuracies, label=label)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.legend(fontsize=16)
    plt.suptitle(title, fontsize=20)



accuracies = []
train_accuracies = []
iterations = 1
for i in range(iterations):
    data = np.load("data/new_100_corner.npy")
    #pivot = np.random.randint(0, int(len(data)/2))
    #train = data[pivot:pivot+int(np.ceil(len(data)/2))]
    #val = np.append(data[:pivot], data[pivot+int(np.ceil(len(data)/2)):], axis=0)
    #train = train.reshape(-1, train.shape[-1])
    #val = val.reshape(-1, val.shape[-1])
    np.random.shuffle(data)
    train = data[:24]
    val = data[24:]
    train = train.reshape(-1, train.shape[-1])
    val = val.reshape(-1, val.shape[-1])
    np.random.shuffle(train)
    np.random.shuffle(val)

    #train stuff
    nr_of_classes = 10
    boys = [10, 20, 30, 50]
    depths = [1, 3, 5, 9, 14]
    epochs = 200
    brains = {}
    for depth in depths:
        for size in boys:
            layers = [data.shape[1]-1]
            for i in range(depth):
                layers.append(size)
            layers.append(nr_of_classes)
            activations = np.full(len(layers), keras.activations.relu)
            activations[-1] = keras.activations.softmax
            network = ANN(layers, activations)
            network.train(train, epochs, val=val)
            brains["D"+str(depth)+"N"+str(size)] = network

    acc_fig = plt.figure(1, figsize=(16, 10))

    #do some saving stuff for plots
    for keys in brains.keys():
        accuracy_plot(brains[keys].model.history.history["val_accuracy"], label=keys, figure=acc_fig)
        accuracies.append(brains[keys].model.history.history["val_accuracy"])
        train_accuracies.append(brains[keys].model.history.history["accuracy"])

np.save("val_dis_raw_accuracies.npy", np.array(accuracies))
np.save("train_dis_raw_accuracies.npy", np.array(train_accuracies))
print(np.mean(accuracies[:,-1]))



#lots of plotting
"""
title = "All in w/o normalization"
load = np.load("normalized/disjunct/accuracies.npy")
labels = ["D1N10", "D1N20", "D1N30", "D1N50",
          "D3N10", "D3N20", "D3N30", "D3N50",
          "D5N10", "D5N20", "D5N30", "D5N50",
          "D9N10", "D9N20", "D9N30", "D9N50",
          "D14N10", "D14N20", "D14N30", "D14N50"]

print(np.argmax(load[:,-1]), np.max(load[:,-1]))
print(load[:,-1]-np.max(load[:,-1]))
exit()

#bar plot
barfig = plt.figure(2, figsize=(10,6))
x = np.array(range(load.shape[0]))
for i in range(4, len(x), 4):
    x[i:] += 1
color = ["blue", "green", "yellow", "red"]*5
plot = plt.bar(x, load[:,-1], color=color)

legend_entries = [Patch(color="white", label="Width"), Patch(color="blue", label="10N"), Patch(color="green", label="20N"), Patch(color="yellow", label="30N"), Patch(color="red", label="50N")]
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', handles=legend_entries, fontsize=16)
plt.xlabel('Depth', fontsize=16)
plt.xticks(x[1::4]+0.5, ['1', '3', '5', '9', '14'], fontsize=16)
plt.ylabel("Validation accuracy", fontsize=16)
barfig.subplots_adjust(right=0.8)
plt.suptitle(title, fontsize=18)
plt.show()

#exit()


#val accuracy plot during training
load = load[12:16] #[[0,2,3]]
labels = np.array(labels)[12:16]#[[0,2,3]]
acc_fig = plt.figure(1, figsize=(16, 10))
for label, models in zip(labels, load):
    accuracy_plot(models, label, acc_fig)
plt.show()
#exit()


accuracies = []
std_devs = []
all_raw_val_acc = np.load("val_all_raw_accuracies.npy")
accuracies.append(np.mean(all_raw_val_acc[:,-1]))
std_devs.append(np.std(all_raw_val_acc[:,-1]))
all_raw_train_acc = np.load("train_all_raw_accuracies.npy")
accuracies.append(np.mean(all_raw_train_acc[:,-1]))
std_devs.append(np.std(all_raw_train_acc[:,-1]))

dis_raw_val_acc = np.load("val_dis_raw_accuracies.npy")
accuracies.append(np.mean(dis_raw_val_acc[:,-1]))
std_devs.append(np.std(dis_raw_val_acc[:,-1]))
dis_raw_train_acc = np.load("train_dis_raw_accuracies.npy")
accuracies.append(np.mean(dis_raw_train_acc[:,-1]))
std_devs.append(np.std(dis_raw_train_acc[:,-1]))


all_norm_val_acc = np.load("val_all_norm_accuracies.npy")
accuracies.append(np.mean(all_norm_val_acc[:,-1]))
std_devs.append(np.std(all_norm_val_acc[:,-1]))
all_norm_train_acc = np.load("train_all_norm_accuracies.npy")
accuracies.append(np.mean(all_norm_train_acc[:,-1]))
std_devs.append(np.std(all_norm_train_acc[:,-1]))


dis_norm_val_acc = np.load("val_dis_norm_accuracies.npy")
accuracies.append(np.mean(dis_norm_val_acc[:,-1]))
std_devs.append(np.std(dis_norm_val_acc[:,-1]))
dis_norm_train_acc = np.load("train_dis_norm_accuracies.npy")
accuracies.append(np.mean(dis_norm_train_acc[:,-1]))
std_devs.append(np.std(dis_norm_train_acc[:,-1]))


print(dis_norm_train_acc.shape, np.mean(dis_norm_train_acc[:,-1]), np.std(dis_norm_train_acc[:,-1]))

#crossvalidation bar plot
barfig = plt.figure(3, figsize=(10,6))
nr_of_bars = 8
group_sizes = 2
x = np.array(range(nr_of_bars))
for i in range(group_sizes, len(x), group_sizes):
    x[i:] += 1
color = ["blue", "green"]*(int(len(x)/group_sizes))
plot = plt.bar(x, accuracies, color=color, yerr=std_devs)

legend_entries = [Patch(color="white", label="Width"),
                  Patch(color="blue", label="Validation"),
                  Patch(color="green", label="Training")]
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', handles=legend_entries, fontsize=16)
plt.xlabel('Methods', fontsize=16)
plt.xticks(x[0::group_sizes]+0.5, ['D1N50 \n all in w/o \n normalization',
                                   'D1N30 \n disjunct w/o \n normalization',
                                   'D1N50 \n all in with \n normalization',
                                   'D1N20 \n disjunct with \n normalization'], fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
barfig.subplots_adjust(right=0.8)
plt.suptitle("Crossvalidation accuracies", fontsize=18)
plt.show()
print(accuracies)
exit()
"""