import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#this script chooses a random number that represents the digit i
#when i==6, for example, the script pulls up a random index with 6 as the label
df = pd.read_csv('mnist_train.csv', header= None)
a = df[df.columns[0]]
label = list((float(i) for i in (a)))
i = 3
# n is our on and off switch for the while loop
n = 0
# it takes the random number chosen by the np.random.choice method and compares it to our desired i
while (n == 0):
        a_choice = np.random.choice(range(n, len(label)))
        if i == label[a_choice]:

            a_pixels = df.loc[a_choice]
            a_pixels = np.array(a_pixels[1:], dtype='uint8')
            a_pixels = a_pixels.reshape((28, 28))
            plt.imshow(a_pixels, cmap='gray')
            plt.title('Label is {i}, Index is {a_choice}'.format(i=i, a_choice=a_choice))
            plt.show()
            n=1
        else:
            n = 0


