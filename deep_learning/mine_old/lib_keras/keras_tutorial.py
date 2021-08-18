from keras.utils import to_categorical
import numpy as np

####################################################

memory_size, memory_index = 100000, 0
n_actions, action_space, a = 4, [0,1,2,3], 1

# One-Hot encoding:

a_indices_one_hot = np.zeros((memory_size, n_actions), dtype=np.ubyte)
a_index = action_space.index(a)
a_indices_one_hot[memory_index, a_index] = 1

a_indices_one_hot = to_categorical(a, n_actions)

####################################################
