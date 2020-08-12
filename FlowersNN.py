"""
Basic NN for determining Red vs Blue flowers from "Beginner Introduction to Neural Networks"
https://www.youtube.com/playlist?list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
"""

import numpy as np
def NN(m1, m2, w1, w2, b):
    z = m1*w1 + m2*w2 + b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def squaredError(p, t):
    return (p - t)**2

w1 = np.random.randn()
w2 = np.random.rand()
b = np.random.rand()

print (w1)
print(w2)
print(b)

print("Predict: "+ str(NN(2, 1, w1, w2, b)))

