import pandas as pd
import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA
from math import e

import math
from Project3_Q1 import transform_data
import sys

# n, 
def initial_probability_distribution(starting_point, data, n):
    min_distance = np.sum(np.square(data - starting_point), axis=1)
    normalized_min_distance = min_distance / np.sum(min_distance);
    gaussian_distribution = np.zeros(n)
    for dim in range(data.shape[1]):
        var = math.pow(np.std(data[:,dim]), 2)
        gaussian_distribution_dim = np.exp(-1*(np.square(data[:,dim] - np.mean(data[:,dim]))/ 2 * var)) / math.sqrt( 2 * math.pi * var)
        gaussian_distribution = gaussian_distribution + gaussian_distribution_dim
    qx = normalized_min_distance/2 + gaussian_distribution/(2 * data.shape[1])
    return qx;

def get_mixing_time(k):
    return int(1 + 8*(math.log(4*k)));

def valid_state(data, y_i, x_0, qx, C):
    d_y_C = np.min(np.sum(np.square(data[y_i] - C), axis=1))
    d_x_C = np.min(np.sum(np.square(data[x_0] - C), axis=1))
    q_y_C = qx[y_i]
    q_x_C = qx[x_0]
    
    if ((d_x_C * q_y_C) == 0):
        chance = 0
    else:
        chance = min(1, (d_y_C * q_x_C)/ (d_x_C * q_y_C))
    rand_num = random.random()
    if (rand_num < chance):
        return True
    return False
    
def approximate_px(data, n, k, p):
    starting_point = data[randint(0, n)]
    mixing_time = get_mixing_time(k);
    qx = initial_probability_distribution(starting_point, data, n);
    prob_distribution = np.cumsum(qx)
    
    C = np.zeros((k, p))
    C[0,:] = data[randint(0, n)]

    number_of_centroids = 0;
    for cen in range(k):
        x_0 = randint(0, n)
        for i in range(mixing_time):
            rand_num = random.random()
            sampled_ind = np.argmax(prob_distribution >= rand_num)
            y_i = sampled_ind
            if valid_state(data, y_i, x_0, qx, C):
                x_0 = y_i
        
        C[number_of_centroids] = data[x_0]
        number_of_centroids = number_of_centroids + 1;
            
    return C

start_time = time.time()
data = transform_data("yelp.csv")

#Normalizing the data
data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
#Batch Size
B = 1000
# Number of Centriods
k_list = [5, 10, 50, 100, 200, 300, 400, 500]
#Iterations
T = 40
#Features
p = data.shape[1]
#Data Points
n = data.shape[0]

mean_for_k = np.zeros((len(k_list)))
min_for_k = np.zeros((len(k_list)))
max_for_k = np.zeros((len(k_list)))

for k_ind in range(len(k_list)):
    k = k_list[k_ind]
    
    # Initialization of the first centroid
    start_centroid_calculation = time.time()    
    C = approximate_px(data, n, k, p);  
    print("Calculated centroids in: {}", time.time() - start_centroid_calculation)
    
    C = C.reshape(k, p, 1)
    # Initialization of Assignmentment 
    A = -1 * np.ones((n, 1))
    loss = np.zeros((T))
    for t in range(T):
        eta = 1/(t+1)
        data_batch = data[t*B: (t+1)*B, :] #B*p
        
        # Assignment Step
        data_dash = data_batch.reshape(B, p, 1) #B*p*1
        data_batch_dash = np.repeat(data_dash, k, axis = 2) #B*p*T
        data_batch_dash = data_batch_dash.transpose(2,1,0) #k*p*B
        
        diff = np.sum(np.square(data_batch_dash - C), axis=1) #k*B 
        A_data_batch = np.argmin(diff, axis=0) #B,
       
        # Loss
        A_centroids = C[A_data_batch]
        distanc = np.sqrt(np.sum(np.square(data_batch - A_centroids[:,:,0]), axis=1))
        loss[t] = np.sum(distanc)

        # Update Step
        C_update = np.unique(A_data_batch, axis=0)
        
        for c in range(C_update.shape[0]):
            c_star = C[C_update[c]] #p,1
            x_star = (data_batch[A_data_batch == C_update[c]]).transpose()
            update = (np.sum((x_star - c_star), axis=1).reshape(p,1))/ x_star.shape[1]
            C[C_update[c]] = (c_star + eta * update)
            
# Testing(For Question 5)
    min_dis = sys.maxsize * np.ones(k)
    max_dis = np.zeros(k)
    avg_dis = np.zeros(k)
    
    ind = np.zeros(k)
    count = np.zeros(k)
    for d in range(n):
        dat = data[d].reshape(p,1)
        distance = np.min(np.sqrt(np.sum(np.square(dat - C), axis=1)))
        centroid = np.argmin(np.sqrt(np.sum(np.square(dat - C), axis=1)))
        avg_dis[centroid] = avg_dis[centroid] + distance
        min_dis[centroid] = min(min_dis[centroid], distance)
        max_dis[centroid] = max(max_dis[centroid], distance)
        ind[centroid] = 1
        count[centroid] = count[centroid] + 1
    
    mean_for_k[k_ind] = np.mean(avg_dis[count > 0] / count[count > 0]) 
    min_for_k[k_ind] = np.mean(min_dis[count > 0]) 
    max_for_k[k_ind] = np.mean(max_dis[count > 0])

plt.plot(np.array(k_list), mean_for_k, 'bo--', linewidth=2, markersize=5, label = "Mean distance")
plt.plot(np.array(k_list), min_for_k, 'go--', linewidth=2, markersize=5, label = "Minimum distance")
plt.plot(np.array(k_list), max_for_k, 'yo--', linewidth=2, markersize=5, label = "Maximum distance")
plt.legend()
plt.show()
print(time.time() - start_time)