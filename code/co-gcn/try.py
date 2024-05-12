# from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
# X=[0,0,0,0,1,1,1,1,2,2,2,2]
# y=[0,0,0,0,1,1,1,1,2,2,2,2]
# sss = StratifiedShuffleSplit(n_splits=4, test_size=0.2,
#                                  random_state=0)
# sss.get_n_splits(X, y)
# for train_index, test_index in sss.split(X, y):
#     X_train,X_test = np.array(X)[train_index],np.array(X)[test_index ]
#     # print(train_index)
#     if 1 in train_index:
#         print('yes')
#     elif 1  in test_index:
#         print('no')
# a=np.ones(5)
# a=a/5
# print(a)
a=[]
a.append([1,2])
a.append([1,2,3,4])
print(a)