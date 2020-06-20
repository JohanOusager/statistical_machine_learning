import numpy as np

a = np.load("new_100_corner_part1.npy")
a = np.append(a, np.load("new_100_corner_part2.npy"), axis=0)
a = np.append(a, np.load("new_100_corner_part3.npy"), axis=0)
a = np.append(a, np.load("new_100_corner_part4.npy"), axis=0)
a = np.append(a, np.load("new_100_corner_part5.npy"), axis=0)
a = np.append(a, np.load("new_100_corner_part6.npy"), axis=0)
np.save("new_100_corner", a)

b = np.load("new_100_corner.npy")
print(b.shape)