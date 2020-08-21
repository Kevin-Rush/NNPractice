"""
Basic NN for determining Red vs Blue flowers from "Beginner Introduction to Neural Networks"
https://www.youtube.com/playlist?list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
"""

import numpy as np
from matplotlib import pyplot as plt
from win32com.client import Dispatch
speak = Dispatch("SAPI.SpVoice")

#each point is leng, width and type (1 for red and 0 for blue)

def NN(m1, m2, w1, w2, b):
    z = m1*w1 + m2*w2 + b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidP (x):
    return sigmoid(x)*(1-sigmoid(x))

def squaredError(p, t):
    cost = (p - t)**2
    slope = 2*(p-t)
    return p-0.1*slope

def which_flower(length, width):
    z = length*w1+width*w2+b
    pred = sigmoid(z)
    print(pred)
    if pred <.5:
        speak.Speak("Blue")
    else:
        speak.Speak("Red")

#flowr = data point = [length, width, colour]
data = [[3, 1.5, 1],
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, 0.5, 1],
        [2, 0.5, 0],
        [5.5, 1, 1],
        [1, 1, 0],]

mysteryFlower = [4.5, 1]

"""
Network Architecture
     o      flower type
    / \     w1, w2, b
   o   o    length, width

"""

w1 = np.random.randn()
w2 = np.random.rand()
b = np.random.rand()

X = np.linspace(-5,5,100)
Y = sigmoidP(X)

#plt.plot(X,Y)
#plt.show()

#scatter data
for i in range(len(data)):
    #plt.axis([0, 6, 0, 2])
    #plt.grid()
    point=data[i]
    colour = "r"
    if point[2] == 0:
        colour = "b"
    #Uncomment to show points
    #plt.scatter(point[0], point[1], c = colour)
#plt.show()

#training loop

learning_rate = 0.2
costs=[]

for i in range(50000):
    ri = np.random.randint(len(data))
    point = data[ri]
    z = point[0]*w1 + point[1]*w2 + b
    pred = sigmoid(z)

    target = point[2]
    cost = (pred - target)**2
    

    #calculating the derivative of the cost_prediction
    dcost_pred = 2*(pred-target)
    #derivative of sigmoid(z) is the sigmoid prime function sigmoidP(x)
    dpred_dz = sigmoidP(z)
    #calculating the derivative in respect to each part of the z
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1
#just building the chain rule
    dcost_dz = dcost_pred*dpred_dz
#all partial derivatives in respect to each parameter
    dcost_dw1 = dcost_dz*dz_dw1 
    dcost_dw2 = dcost_dz*dz_dw2 
    dcost_db = dcost_dz*dz_db
#change all the parameters in accordance to their error cost
    w1 = w1 - learning_rate*dcost_dw1
    w2 = w2 - learning_rate*dcost_dw2
    b = b - learning_rate*dcost_db
#calculating cost
    if i % 100 == 0:
        cost_sum = 0
        for j in range(len(data)):
            point = data[ri]
            z = point[0]*w1 + point[1]*w2 + b
            pred = sigmoid(z)
            target = point[2]
            cost_sum +=(pred - target)**2
        
        costs.append(cost_sum/len(data))

#Uncomment to show cost graph
#plt.plot(costs)        
#plt.show()

#see all predictions on known data
for i in range(len(data)):
    point = data[i]
    print(point)
    z = point[0]*w1 + point[1]*w2 + b
    pred = sigmoid(z)
    print("Pred: {}".format(pred))

#voice functionality
which_flower(3, 1)