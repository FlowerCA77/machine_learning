import numpy as np
from matplotlib import pyplot as plt

def create_dataSet() -> tuple:
    group = np.array(
        [[1.0, 1.1],
         [1.0, 1.0],
         [0  , 0  ],
         [0  , 0.1]]
    )
    features = np.array(['A', 'A', 'B', 'B'])
    return group, features

def show_dataSet(ds:tuple):
    print(f'dataset =\n{ds}\n')
    group, features = ds[0], ds[1]
    print(f'group =\n{group}\nfeatures = \n{features}\n')
    print(f'group num = {group.shape[0]}, features num = {np.size(features)}\n')

def plot_dataSet(ds:tuple):
    group, features = ds[0], ds[1]
    features = np.pad(
        features,
        (0, max(0, group.shape[0]-np.size(features))),
        mode="constant",
        constant_values=''
    )
    plt.scatter(group[:,0], group[:,1])
    for i in range(group.shape[0]):
        plt.text(group[i,0], group[i,1], features[i])
    plt.title(f'kNN dataset')
    plt.grid()
    plt.show()