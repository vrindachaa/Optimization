import pandas as pd
import numpy as np

#prepping data
test = pd.read_csv('df_xy_test.csv', header = None)
weights = pd.read_csv('weights011.csv', header=None, dtype= float)
test = test.drop(test.columns[0], axis= 1)
test = test.drop(test.index[0], axis= 0)
weights = weights.drop(weights.columns[0], axis= 1)
weights = weights.drop(weights.index[0], axis= 0)
test_labels = test.iloc[:,0]
test = test.drop(test.columns[0], axis= 1)

def classify(test_set,weights, test_label_list):
    test_label_list = list(-1 if (x == 4) else 1 for x in test_label_list)
    b_pred = np.sign(np.dot(test_set,weights))
    n =0
    for i in range(len(test_label_list)):
        if b_pred[i] == test_label_list[i]:
            n+=1
    return n
print(classify(test, weights, test_labels))