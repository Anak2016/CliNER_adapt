# import collections
#
# # token_count = collections.defaultdict(int)  # initialized by a function
# token_count = collections.defaultdict(lambda: 0)  # initialized by a function
# tokens = ['follow', 'follow','follow','follow', 'hi', 'you']
#
# for token_i in tokens:
#     token_count[token_i] += 1
#
# print(token_count['hi'])

import numpy as np
token_features= [5,6]

features_as_array = np.array(token_features, dtype=np.dtype('int32'))
print(features_as_array)
# features_as_array = features_as_array.reshape((features_as_array.shape[0], 1))
features_as_array = features_as_array.reshape((1, features_as_array.shape[0]))
print(features_as_array)
features_as_array = np.transpose(features_as_array)
print(features_as_array)

