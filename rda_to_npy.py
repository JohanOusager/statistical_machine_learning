import pyreadr
import numpy as np
import argparse

#read nested RDA (list of objects/matrices)
def read_all_in(path_dot_rda, shape, verbose=False):
    data, data_size = read_Rda(path_dot_rda, verbose)
    data = np.squeeze(data)

    data = restore3D_Rda(data, shape) #[10, 4000, 325])
    #data = np.reshape(data, [40000, 325])
    return data, data.shape


#read dumb ugly R data file
def read_Rda(path_to_Rda, verbose=False):
    Rdata = pyreadr.read_r(path_to_Rda) # also works for Rds
    while len(Rdata.keys()) == 1:
        keys = np.array(list(Rdata.keys()))
        Rdata = Rdata[keys[0]]
    npdata = np.array(Rdata, dtype=np.double)
    data_size = npdata.shape
    if verbose:
        print("Sample size is ", npdata.shape)
    return npdata, data_size


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


def main(input_file, shape):
    if shape is None or len(shape) == 0:
        data, shape = read_Rda(input_file)
        np.save(input_file[:-4], data)
        print(input_file[:-4], ".npy saved with shape ", shape)
    else:
        data, shape = read_all_in(input_file, shape)
        np.save(input_file[:-4], data)
        print(input_file[:-4], ".npy saved with shape ", shape)



print("Starting")
parser = argparse.ArgumentParser(description='convert .RDS or .RDA to .npy')
parser.add_argument('-R_file', help='input file (RDA or RDS)')
parser.add_argument('-shape', nargs='+', type=int, help='shape of the data')
args = parser.parse_args()
main(args.R_file, args.shape)

