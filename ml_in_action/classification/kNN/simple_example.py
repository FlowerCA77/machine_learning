import numpy as np

import gen_data as kNN_gd
import classify as kNN_classify

ds = kNN_gd.create_dataSet()
group, features = ds[0], ds[1]

X = np.array([0.7, 0.6])
F = kNN_classify.classify(X, group, features, 5)

group = np.append(group, np.array([X]), axis=0)
features = np.append(features, np.array([F]), axis=0)

kNN_gd.plot_dataSet((group, features))
