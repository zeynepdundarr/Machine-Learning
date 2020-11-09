#!/usr/bin/env python
# coding: utf-8

# In[6]:





# In[7]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math 

np.random.seed(421)
# mean parameters


class_means = np.array([[+0.0, +2.5],
                        [-2.5, -2.0],
                        [+2.5, -2.0]])
# covariance parameters
class_covariances = np.array([[[+3.2, +0.0], 
                               [+0.0, +1.2]],
                              [[+1.2, -0.8], 
                               [-0.8, +1.2]],
                              [[+1.2, +0.8], 
                               [+0.8, +1.2]]])
# sample sizes
class_sizes = np.array([120, 90, 90])
N = 300


# In[8]:


# generate random samples
points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])
points = np.concatenate((points1, points2, points3), axis=0)
X = np.vstack((points1, points2, points3))# again
y_truth = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))

#sample_mean1 = np.array([np.mean(points,axis=0)]).transpose() #2x1 matrix
sample_mean1 = np.mean(points1,axis=0).reshape(2,1)
sample_mean2 = np.mean(points2,axis=0).reshape(2,1)
sample_mean3 = np.mean(points3,axis=0).reshape(2,1)
sample_means = np.array([sample_mean1, sample_mean2, sample_mean3])

sample_cov1 = np.cov(points1.transpose())
sample_cov2 = np.cov(points2.transpose())
sample_cov3 = np.cov(points3.transpose())
sample_covs = np.array([sample_cov1,sample_cov2,sample_cov3])

priors = [np.mean(y_truth == c+1) for c in range(3)]

data1 = points1.transpose()
data2 = points2.transpose()
data3 = points3.transpose()

plt.figure(figsize = (6, 6))
plt.plot(data1[0],data1[1],'.r')
plt.plot(data2[0],data2[1],'.g')
plt.plot(data3[0],data3[1],'.b')

plt.xlabel('x1')
plt.xlabel('x2')

#test for Q2
print("---Estimated Means---")
print(sample_means)
print("\n---Estimated Covariances---")
print(sample_covs)
print("\n---Estimated Priors---")
print(priors)


# In[9]:


K = 3
W = [-1/2*np.linalg.inv(sample_covs[c]) for c in range(K)]
w = [np.matmul(np.linalg.inv(sample_covs[c]),sample_means[c]) for c in range(K)]
w0 =[-1/2*np.matmul(sample_means[c].transpose(),np.matmul(np.linalg.inv(sample_covs[c]),sample_means[c]))
     -1/2*np.log(np.linalg.det(sample_covs[c])) + np.log(priors[c]) for c in range(K)]  


# In[10]:


y_label = []
datapoints = np.concatenate((points1,points2,points3),axis=0)

def score(x,c):
    x = np.array(x).reshape(2,1)
    gcx = np.matmul(x.T,np.matmul(W[c],x)) + np.matmul(w[c].T,x) + w0[c] 
    return gcx[0][0]
          
for x in datapoints:
    x = x.reshape(2,1)
    gcx = [score(x,c) for c in range(K)]
    y_label.append(np.argmax(gcx)+1)


# In[11]:


pd.crosstab(pd.Series(np.array(y_label)),pd.Series(y_truth),
            rownames=["ypredicted"],colnames=["ytruth"],dropna=False) #icine array de aliyor


# In[12]:


#evaluate discriminant function on a grid
x1_interval = np.linspace(-6, +6, 80)
x2_interval = np.linspace(-6, +6, 80)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), 3))
K = 3

for i, x2 in enumerate(x2_interval):
    for j, x1 in enumerate(x1_interval):
        for c in range(K):
            discriminant_values[i,j,c] = score([x1,x2],c)    


A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]


A[(A < B) & (A < C)] = np.nan #logic behind
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C

plt.figure(figsize = (10, 10))
plt.plot(datapoints[y_truth == 1, 0], datapoints[y_truth == 1, 1], "r.", markersize = 10)
plt.plot(datapoints[y_truth == 2, 0], datapoints[y_truth == 2, 1], "g.", markersize = 10)
plt.plot(datapoints[y_truth == 3, 0], datapoints[y_truth == 3, 1], "b.", markersize = 10)
plt.plot(datapoints[y_label != y_truth, 0], datapoints[y_label != y_truth, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contourf(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1] ,colors=["#EE6C4D","#98C1D9"],levels = 0)
plt.contourf(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2] ,colors=["#3D5A80","#98C1D9"],levels = 0)
plt.contourf(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,2] ,colors=["#3D5A80","#EE6C4D"],levels = 0)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()



#%


# In[12]:





# In[12]:





# In[12]:





# In[12]:




