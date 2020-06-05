import pandas as pd
import numpy as np
import random
from random import randint
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
import sys
from Project3_Q1 import transform_data

start_time = time.time()
data = transform_data("yelp.csv")

#Normalizing the data
data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
#Batch Size
B = 1000
# Number of Centriods
k_list = [5, 10, 20, 50, 100, 150, 250, 500]
#k_list = [20]
#Iterations
T = 40
#t1 = range(1, 41)
#Features
p = data.shape[1]
#Data Points
n = data.shape[0]

#Params for Batch Size Tuning
#batch_size_list = [5, 10, 50, 100, 500, 1000, 5000]
#k_list = [10]
mean_for_k = np.zeros((len(k_list)))
min_for_k = np.zeros((len(k_list)))
max_for_k = np.zeros((len(k_list)))

for k_ind in range(len(k_list)):
    k = k_list[k_ind]
    
    # Initialization of the first centroid
    C = np.zeros((k, p))
    C[0,:] = data[randint(0, n)]
    
    min_distance = np.sqrt(np.sum(np.square(data - C[0,:]), axis=1))
    start_centroid_calculation = time.time()    
    for centroid_num in range(k - 1):
        min_with_last_centroid = np.sqrt(np.sum(np.square(data - C[centroid_num,:]), axis=1))
        min_distance = np.minimum(min_distance, min_with_last_centroid)
        normalized_min_distance = min_distance / np.sum(min_distance);
        prob_distribution = np.cumsum(normalized_min_distance)
        rand_num = random.random()
        sampled_ind = np.argmax(prob_distribution >= rand_num)
        C[centroid_num + 1, :] = data[sampled_ind];
    
    print("Calculated centroids in: {}", time.time() - start_centroid_calculation)
    
    C = C.reshape(k, p, 1)
    # Initialization of Assignmentment 
    A = -1 * np.ones((n, 1))
    loss = np.zeros((T))
    #for B in batch_size_list:
    for t in range(T):
        eta = 1/(t+1)
        data_ind = np.random.choice(data.shape[0], B, replace = False)
        data_batch = data[data_ind] #B*p
        
        # Assignment Step
        data_dash = data_batch.reshape(B, p, 1) #B*p*1
        data_batch_dash = np.repeat(data_dash, k, axis = 2) #B*p*T
        data_batch_dash = data_batch_dash.transpose(2,1,0) #k*p*B
        
        diff = np.sum(np.square(data_batch_dash - C), axis=1) #k*B 
        A_data_batch = np.argmin(diff, axis=0) #B,
        
        # Loss
        A_centroids = C[A_data_batch]
        distanc = np.sqrt(np.sum(np.square(data_batch - A_centroids[:,:,0]), axis=1))
        loss[t] = np.sum(distanc) / B
       
         # Update Step
        C_update = np.unique(A_data_batch, axis=0)
        loss_for_cen = np.zeros(C_update.shape[0])
        for c in range(C_update.shape[0]):
            c_star = C[C_update[c]] #p,1
            x_star = (data_batch[A_data_batch == C_update[c]]).transpose()
            loss_for_cen[c] = np.mean(np.sqrt(np.sum(np.square(x_star - c_star), axis=0)))
            update = (np.sum((x_star - c_star), axis=1).reshape(p,1))/ x_star.shape[1]
            C[C_update[c]] = (c_star + eta * update)
        
        # Loss
        loss[t] = np.mean(loss_for_cen)
   #plt.plot(t1,loss,label="B = " + str(B))
#Plotting Number of Iterations vs Loss for different Batch size
# =============================================================================
# plt.xlabel("Number of Iterations(t)")
# plt.ylabel("Loss")
# plt.title("K-Means++: Number of Iterations vs Loss for different Batch size") 
# plt.legend(bbox_to_anchor=(1.1, 1.05))
# plt.legend()
# plt.show()  
   
#Plotting Number of Iterations vs Loss for different k
# =============================================================================           
#    print("K: {}, Loss: {}", k, loss)
#    plt.plot(range(T), loss,  linewidth=2, markersize=5, label = k)
#    plt.xlabel("Number of Iterations")
#    plt.ylabel("Loss")
#    plt.title("Number of Iterations vs Loss for different k")
#    plt.legend()
#    plt.show()
    
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
#figure()
plt.plot(np.array(k_list), mean_for_k, 'bo--', linewidth=2, markersize=5, label = "Mean")
plt.plot(np.array(k_list), min_for_k, 'go--', linewidth=2, markersize=5, label = "Minimum")
plt.plot(np.array(k_list), max_for_k, 'yo--', linewidth=2, markersize=5, label = "Maximum")
plt.xlabel("Number of Centroids(k)")
plt.ylabel("Distance")
plt.title("K-Means++: Number of Centroids vs Distance")
plt.legend()
plt.show()
print(time.time() - start_time)