import sys, os
from NNTraining import NNTraining

if len(sys.argv)<3:
    print('Usage:', sys.argv[0], '<batch_size> <num_neurons> <path_to_NPZ_file>')
    exit()

bs = int(sys.argv[1])
nn = int(sys.argv[2])
npz_path = sys.argv[3]

NNT = NNTraining(num_neurons=nn, batch_size=bs)

NNT.train_on_npz(npz_path, wave_range=(4199, 5800), validation_fraction=0.1)




