#!/usr/bin/env python
# coding: utf-8

# In[233]:


import numpy as np
#from  Utils import *
from matplotlib import image
from skimage import color
from skimage import io
from Calculate_Features2 import *

from statsmodels.stats.contingency_tables import mcnemar
import os

from sklearn import metrics, svm
import image_slicer
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV


# change into 3D
# binned_img daki distance a gore secilen pairlardan kac tane oldugunu co_occur un pair1 ve pair2 indexlerine koy
def calculateCooccurrenceMatrix(binned_img, binNumber, di, dj):
    binned_img = findBin(binned_img, binNumber)
    max_i = binNumber

    co_occur = [[0 for _ in range(max_i+1)] for _ in range(max_i+1)]
    co_occur = np.array(co_occur)

    for x1 in range(len(binned_img)):
        for y1 in range(len(binned_img[0])):

            if di + x1 >= 0 and dj + y1 >=0 and di+x1 < len(binned_img) and dj+y1 < len(binned_img[0]):
                x2 = x1+di
                y2 = y1+dj
                co_occur[int(binned_img[x1,y1]), int(binned_img[x2,y2])] += 1
            else:
                # switch other column
                if dj+y1<0:
                     continue
                # break column loop, continue with row loop
                if di+x1<0:
                      break
                if di+x1>max_i:
                      break
                if dj+y1>max_i:
                      break

    return co_occur

def calculateAccumulatedCooccurrenceMatrix(grayImg, binNumber, d):
    dlist = [(d, 0), (d, d), (0, d),(-d, d), (-d, 0), (-d, -d), (0, -d), (d, -d)]
    shape = calculateCooccurrenceMatrix(grayImg,binNumber,dlist[0][0],dlist[0][1]).shape
    sum_co_occur = np.zeros(shape, dtype=int)
    for i in range(1,len(dlist)):
        sum_co_occur = sum_co_occur + calculateCooccurrenceMatrix(grayImg,binNumber,dlist[i][0],dlist[i][1])
    return sum_co_occur


def findBin(gray_img, binno):
    temp = gray_img.copy()
    bins = np.arange(0, 1, 1/binno, dtype=float)
    for row_ind in range(len(temp)):
        an_array = temp[row_ind]
        bin_indices = np.digitize(an_array, bins)
        temp[row_ind] = bin_indices

    return temp

def processImg(filepath,binno,d):
    img = io.imread(filepath)
    grayImg = color.rgb2gray(img)
    an_array = calculateAccumulatedCooccurrenceMatrix(grayImg, binno, d)    
    normal_array = (an_array - np.mean(an_array))/np.std(an_array)

#     norm = np.linalg.norm(an_array)
#     normal_array = an_array / norm
    return normal_array


#current directory edit
def getFeatures(name,path):
    binno = 8
    d = 10
    curr_dir = os.getcwd() 
    
    if name == "itr":     
        filepath = path + "/tr"
        dataset_tr = np.zeros(186 * 6).reshape((186, 6))
        dataset = dataset_tr
        start = 1
        size = 186
        end = start + size
        
    elif name == "itr1":     
        filepath = path + "/tr"
        dataset_tr = np.zeros(60 * 6).reshape((60, 6))
        dataset = dataset_tr
        start = 1
        size = 60
        end = start + size
        
    elif name == "itr2":     
        filepath = path + "/tr"
        dataset_tr = np.zeros(88 * 6).reshape((88, 6))
        dataset = dataset_tr
        start = 61
        size = 88
        end = start + size
        
    elif name == "itr3":     
        filepath = path + "/tr"
        dataset_tr = np.zeros(38 * 6).reshape((38, 6))
        dataset = dataset_tr
        start = 149
        size = 38
        end = start + size    
        
    elif name == "btr":     
        filepath = path + "/tr"
        dataset_tr = np.zeros(264 * 6).reshape((264, 6))
        dataset = dataset_tr
        start = 1
        size = 264
        end = start + size
        
    elif name == "btr1":     
        filepath = path + "/tr"
        dataset_tr = np.zeros(88 * 6).reshape((88, 6))
        dataset = dataset_tr
        start = 1
        size = 88
        end = start + size
        
    elif name == "btr2":     
        filepath = path + "/tr"
        dataset_tr = np.zeros(88 * 6).reshape((88, 6))
        dataset = dataset_tr
        start = 89
        size = 88
        end = start + size
        
    elif name == "btr3":     
        filepath = path + "/tr"
        dataset_tr = np.zeros(88 * 6).reshape((88, 6))
        dataset = dataset_tr
        start = 177
        size = 88
        end = start + size      

    elif name == "ts":     
        filepath = path + "/ts"
        dataset_ts = np.zeros(144 * 6).reshape((144, 6))
        dataset = dataset_ts
        start = 1
        size = 144
        end = start + size
        
    elif name == "ts1":     
        filepath = path + "/ts"
        dataset_ts = np.zeros(48 * 6).reshape((48, 6))
        dataset = dataset_ts
        start = 1
        size = 48
        end = start + size
        
    elif name == "ts2":     
        filepath = path + "/ts"
        dataset_ts = np.zeros(57 * 6).reshape((57, 6))
        dataset = dataset_ts
        start = 49
        size = 57
        end = start + size
        
    elif name == "ts3":     
        filepath = path + "/ts"
        dataset_ts = np.zeros(39 * 6).reshape((39, 6))
        dataset = dataset_ts
        start = 106
        size = 39
        end = start + size
        
    elif  name == "sub":     
        print("sub")
        filepath = curr_dir + "/dataset2/sliced/" 
        dataset_sub = np.zeros(16*6).reshape((16,6))
        dataset = dataset_sub
        
    
    k = 0
    if name != "sub":
        for i in range(start, end):#len(os.listdir(folder))):
            jpg_num = i
            filename_path = filepath+str(jpg_num)+".jpg"
            print("filepath 2: ", filename_path)
            normal_array = processImg(filename_path, binno, d)
            ### ek
            normalized = True
            feat_arr = calculateCooccurrenceFeatures(normal_array, normalized)
            # ### ek
            dataset[k] = feat_arr
            k += 1

        return dataset                 

    elif name == "sub":
                        
        for filename in os.listdir(filepath):
        
            filename_path = filepath + filename
            print("filepath 3: ", filename_path)
            normal_array = processImg(filename_path, binno, d)
            #feat_arr = dummyFeature(normal_array,i)

            ### ek
            normalized = True
            feat_arr = calculateCooccurrenceFeatures(normal_array, normalized)
            # ### ek
            dataset[k] = feat_arr
            k += 1
         
        return np.mean(dataset,axis=0)               

def getSlicedFeatures(path, name):
    
    size = 0
    
    if name == "ts":   
        size = 144
    elif name == "ts1":
        size = 48 
    elif name == "ts2":
        size = 57
    elif name == "ts3":
        size = 39
    elif name == "tr":
        size = 264
    elif name == "tr1":
        size = 88
    elif name == "tr2":
        size = 88   
    elif name == "tr3":
        size = 88     
    
           
    curr_dir = os.getcwd() 
    folder_path = path + "/"
    save_path = curr_dir + "/dataset2/sliced/"
    
    N = 4
    dataset_sub = np.zeros(size * 6).reshape((size, 6))
    i = 0
    for filename in os.listdir(folder_path):
        print("filename is: "+ filename)
        tiles = image_slicer.slice(folder_path+filename, N, save=False)
        image_slicer.save_tiles(tiles, directory= save_path)
        dataset_sub[i] = getFeatures("sub", name)

    return dataset_sub


def mc_nemar(y_rbf, y_linear):

    table = pd.crosstab(pd.Series(y_rbf),pd.Series(y_linear),
                rownames=["y_rbf"],colnames=["y_linear"],dropna=False) #icine array de aliyor
    # calculate mcnemar test
    result = mcnemar(table, exact=True)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue >= alpha:
        return 0
    else:
        return 1

def eval_mc_nemar(y_rbf, y_linear, cl):
    print()
    print()
    print("Class: ",cl)
    if mc_nemar(y_rbf, y_linear) != 0:
        print("statistically insignificant results in train prediction.")
    else:    
        print("statistically significant results in train prediction.")
        
    if mc_nemar(y_rbf, y_linear) != 0:
        print("statistically insignificant results in test prediction.")
    else:    
        print("statistically significant results in test prediction.")  
        print()
        print()
    


# In[220]:


#path is generalized
curr_dir = os.getcwd() 

balanced_path_train_1 =  curr_dir +"/dataset2/train_balanced/train_balanced_1"
balanced_path_train_2 =  curr_dir +"/dataset2/train_balanced/train_balanced_2"
balanced_path_train_3 =  curr_dir +"/dataset2/train_balanced/train_balanced_3"
balanced_path_train_all =  curr_dir +"/dataset2/train_balanced/train_balanced_all"

imbalanced_path_train_1 =  curr_dir +"/dataset2/train_imbalanced/train_imbalanced_1"
imbalanced_path_train_2 =  curr_dir +"/dataset2/train_imbalanced/train_imbalanced_2"
imbalanced_path_train_3 =  curr_dir +"/dataset2/train_imbalanced/train_imbalanced_3"
imbalanced_path_train_all =  curr_dir +"/dataset2/training"

test_all =  curr_dir +"/dataset2/test/test_all"
test_1 =  curr_dir +"/dataset2/test/test_1"
test_2 =  curr_dir +"/dataset2/test/test_2"
test_3 =  curr_dir +"/dataset2/test/test_3"

balanced_path_label_1 =  curr_dir +"/dataset2/labels/balanced_training_labels_1.txt"
balanced_path_label_2 =  curr_dir +"/dataset2/labels/balanced_training_labels_2.txt"
balanced_path_label_3 =  curr_dir +"/dataset2/labels/balanced_training_labels_3.txt"
balanced_path_label_all =  curr_dir +"/dataset2/labels/balanced_training_labels_all.txt"

imbalanced_path_label_1 =  curr_dir +"/dataset2/labels/imbalanced_train_labels_1.txt"
imbalanced_path_label_2 =  curr_dir +"/dataset2/labels/imbalanced_train_labels_2.txt"
imbalanced_path_label_3 =  curr_dir +"/dataset2/labels/imbalanced_train_labels_3.txt"
imbalanced_path_label_all =  curr_dir +"/dataset2/labels/imbalanced_training_labels_all.txt"

test_all_label =  curr_dir +"/dataset2/labels/test_labels_all.txt"
test_1_label =  curr_dir +"/dataset2/labels/test_label_1.txt"
test_2_label =  curr_dir +"/dataset2/labels/test_label_2.txt"
test_3_label =  curr_dir +"/dataset2/labels/test_label_3.txt"


# In[221]:


# # part 1

balanced_path_train_1 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/train_balanced/train_balanced_1"
balanced_path_train_2 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/train_balanced/train_balanced_2"
balanced_path_train_3 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/train_balanced/train_balanced_3"
balanced_path_train_all = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/train_balanced/train_balanced_all" 

imbalanced_path_train_1 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/train_imbalanced/train_imbalanced_1"
imbalanced_path_train_2 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/train_imbalanced/train_imbalanced_2"
imbalanced_path_train_3 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/train_imbalanced/train_imbalanced_3"
imbalanced_path_train_all = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/training"

test_all = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/test/test_all"
test_1 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/test/test_1"
test_2 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/test/test_2"
test_3 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/test/test_3"

balanced_path_label_1 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/balanced_training_labels_1.txt"
balanced_path_label_2 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/balanced_training_labels_2.txt"
balanced_path_label_3 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/balanced_training_labels_3.txt"
balanced_path_label_all = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/balanced_training_labels_all.txt"

imbalanced_path_label_1 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/imbalanced_train_labels_1.txt"
imbalanced_path_label_2 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/imbalanced_train_labels_2.txt"
imbalanced_path_label_3 = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/imbalanced_train_labels_3.txt"
imbalanced_path_label_all = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/imbalanced_training_labels_all.txt"

test_all_label = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/test_labels_all.txt"
test_1_label = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/test_label_1.txt"
test_2_label = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/test_label_2.txt" 
test_3_label = "/Users/zeynepdundar/Desktop/Comp448_HW2/dataset2/labels/test_label_3.txt"


# In[222]:


## read labels

test_1_label = np.genfromtxt(test_1_label, dtype=int)
test_2_label = np.genfromtxt(test_2_label, dtype=int)
test_3_label= np.genfromtxt(test_3_label, dtype=int)
test_all_label = np.genfromtxt(test_all_label, dtype=int)

balanced_label_1 = np.genfromtxt(balanced_path_label_1 , dtype=int)
balanced_label_2 = np.genfromtxt(balanced_path_label_2 , dtype=int)
balanced_label_3 = np.genfromtxt(balanced_path_label_3 , dtype=int)
balanced_all_label= np.genfromtxt(balanced_path_label_all , dtype=int)

imbalanced_label_1 = np.genfromtxt(imbalanced_path_label_1 , dtype=int)
imbalanced_label_2 = np.genfromtxt(imbalanced_path_label_2 , dtype=int)
imbalanced_label_3 = np.genfromtxt(imbalanced_path_label_3 , dtype=int)
imbalanced_label_all = np.genfromtxt(imbalanced_path_label_all , dtype=int)


# In[223]:


X_train_i = getFeatures("itr",imbalanced_path_train_all)
X_train_b = getFeatures("btr",balanced_path_train_all)
X_test_all = getFeatures("ts",test_all)

X_test_1 = getFeatures("ts1",test_1)
X_test_2 = getFeatures("ts2",test_2)
X_test_3 = getFeatures("ts3",test_3)

X_train_1_b = getFeatures("btr1",balanced_path_train_1)
X_train_2_b = getFeatures("btr2",balanced_path_train_2)
X_train_3_b = getFeatures("btr3",balanced_path_train_3)

X_train_1_i = getFeatures("itr1",imbalanced_path_train_1)
X_train_2_i = getFeatures("itr2",imbalanced_path_train_2)
X_train_3_i= getFeatures("itr3",imbalanced_path_train_3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[241]:


def test(X_train,X_test,y_train,y_test,clf, cl): 
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
#     print("y_pred_test size: ", len(y_pred_test))
#     print("y_test size: ", len(y_test))
#     print("y_train size: ", len(y_train))
    print("y_pred test len is:", len(y_pred_test))
    print("y_test len is:", len(y_test))
    print("Class: ",cl) 
    print("Accuracy train: %.3f" % metrics.accuracy_score(y_train, y_pred_train))
    print("Accuracy test: %.3f" % metrics.accuracy_score(y_test, y_pred_test))
    
    print("-------------------------------------")
    return y_pred_train, y_pred_test

    
def train(X_train,y_train,kernel):
    
    if kernel == "rbf" or kernel == "rbf_s" or kernel == "0b - rbf" or kernel == "1b - rbf" or kernel == "2b - rbf" or kernel == "3b - rbf":
        param_list = [{'kernel': ['rbf'], 'gamma': [1, 10, 1e-1, 1e-2,1e-3, 1e-4,1e-5], 'C': [1, 10, 100, 1000,5000,50000, 5000000, 50000000]}]
    elif kernel == "linear" or kernel == "linear_s" or kernel == "0b - linear" or kernel == "1b - linear" or kernel == "2b - linear" or kernel == "3b - linear":
        param_list = [{'kernel': ['linear'], 'C': [10, 100, 1000,5000,50000, 5000000, 50000000]}]
    else:
        param_list = [{'kernel': ['rbf'], 'gamma': [10, 1,1e-1, 1e-2,1e-3, 1e-4,1e-5], 'C': [1, 10, 100, 1000,5000,50000, 5000000, 50000000]},
                      {'kernel': ['linear'], 'C': [1, 10, 100, 1000,5000,50000, 5000000, 50000000]}]

    clf = GridSearchCV(svm.SVC(), param_list)
    clf.fit(X_train, y_train)
    
    print("Selected kernel is:", kernel) 
    print("Best parameter is", clf.best_params_)
    return clf




# In[244]:


clf_linear = train(X_train_b, balanced_all_label, "linear") 
y_train_pred_linear_all, y_test_pred_linear_all = test(X_train_b, X_test_all, balanced_all_label, test_all_label,clf_linear, "0b - linear")
y_train_pred_linear_1, y_test_pred_linear_1 = test(X_train_1_b, X_test_1, balanced_label_1, test_1_label, clf_linear,"1b - linear")  
y_train_pred_linear_2, y_test_pred_linear_2 = test(X_train_2_b, X_test_2, balanced_label_2, test_2_label, clf_linear,"2b - linear")
y_train_pred_linear_3, y_test_pred_linear_3 = test(X_train_3_b, X_test_3, balanced_label_3, test_3_label, clf_linear,"3b - linear")


clf_rbf = train(X_train_b, balanced_all_label, "rbf")
y_train_pred_rbf_all, y_test_pred_rbf_all = test(X_train_b, X_test_all, balanced_all_label, test_all_label, clf_rbf,"0b - rbf ")
y_train_pred_rbf_1, y_test_pred_rbf_1 = test(X_train_1_b, X_test_1, balanced_label_1, test_1_label, clf_rbf,"1b - rbf")
y_train_pred_rbf_2, y_test_pred_rbf_2 = test(X_train_2_b, X_test_2, balanced_label_2, test_2_label, clf_rbf,"2b - rbf")
y_train_pred_rbf_3, y_test_pred_rbf_3 = test(X_train_3_b, X_test_3, balanced_label_3, test_3_label, clf_rbf, "3b - rbf")




# for imbalanced no need to obtain y_test_pred y_train_pred results
clf_linear = train(X_train_i, imbalanced_label_all, "linear") 
test(X_train_i, X_test_all, imbalanced_label_all, test_all_label,clf_linear, "0i' - linear")
test(X_train_1_i, X_test_1, imbalanced_label_1, test_1_label, clf_linear,"1i - linear")  
test(X_train_2_i, X_test_2, imbalanced_label_2, test_2_label, clf_linear,"2i - linear")
test(X_train_3_i, X_test_3, imbalanced_label_3, test_3_label, clf_linear,"3i - linear")

clf_rbf = train(X_train_i, imbalanced_label_all, "rbf") 
test(X_train_i, X_test_all, imbalanced_label_all, test_all_label,clf_rbf, "0i - rbf")
test(X_train_1_i, X_test_1, imbalanced_label_1, test_1_label, clf_rbf,"1i - rbf")  
test(X_train_2_i, X_test_2, imbalanced_label_2, test_2_label, clf_rbf,"2i - rbf")
test(X_train_3_i, X_test_3, imbalanced_label_3, test_3_label, clf_rbf,"3i - rbf")



# In[228]:


## data for part 3

X_train_all_s = getSlicedFeatures(balanced_path_train_all,"tr")
X_test_all_s = getSlicedFeatures(test_all,"ts")

X_test_1_s = getSlicedFeatures(test_1,"ts1")
X_test_2_s = getSlicedFeatures(test_2,"ts2")
X_test_3_s = getSlicedFeatures(test_3,"ts3")

X_train_1_s = getSlicedFeatures(balanced_path_train_1,"tr1")
X_train_2_s = getSlicedFeatures(balanced_path_train_2,"tr2")
X_train_3_s = getSlicedFeatures(balanced_path_train_3,"tr3")



# In[245]:



clf_linear_s = train(X_train_all_s, balanced_all_label, "linear_s") 
y_train_pred_linear_all_s, y_test_pred_linear_all_s = test(X_train_all_s, X_test_all_s, balanced_all_label, test_all_label,clf_linear_s,                                                "0b - linear_s")
y_train_pred_linear_1_s, y_test_pred_linear_1_s = test(X_train_1_s, X_test_1_s, balanced_label_1, test_1_label, clf_linear_s,
                                                 "1b - linear_s")  # add x train1 and y_train1
y_train_pred_linear_2_s, y_test_pred_linear_2_s = test(X_train_2_s, X_test_2_s, balanced_label_2, test_2_label, clf_linear_s,
                                                 "2b - linear_s")
y_train_pred_linear_3_s, y_test_pred_linear_3_s = test(X_train_3_s, X_test_3_s, balanced_label_3, test_3_label, clf_linear_s,
                                                 "3b - linear_s")


clf_rbf_s = train(X_train_all_s, balanced_all_label, "rbf_s")
y_train_pred_rbf_all_s, y_test_pred_rbf_all_s = test(X_train_all_s, X_test_all_s, balanced_all_label, test_all_label, clf_rbf_s,
                                            "0b - rbf_s")
y_train_pred_rbf_1_s, y_test_pred_rbf_1_s = test(X_train_1_s, X_test_1_s, balanced_label_1, test_1_label, clf_rbf_s,
                                           "1b - rbf_s")
y_train_pred_rbf_2_s, y_test_pred_rbf_2_s = test(X_train_2_s, X_test_2_s, balanced_label_2, test_2_label, clf_rbf_s,
                                           "2b - rbf_s")
y_train_pred_rbf_3_s, y_test_pred_rbf_3_s = test(X_train_3_s, X_test_3_s, balanced_label_3, test_3_label, clf_rbf_s,
                                           "3b - rbf_s")



train_pair_1_linear = np.array([y_train_pred_linear_1_s, y_train_pred_linear_1, "tr1 - linear"])
train_pair_2_linear = np.array([y_train_pred_linear_2_s, y_train_pred_linear_2, "tr2 - linear"])
train_pair_3_linear = np.array([y_train_pred_linear_3_s, y_train_pred_linear_3, "tr3 - linear"])
train_pair_all_linear = np.array([y_train_pred_linear_all_s, y_train_pred_linear_all, "tr - linear"])


test_pair_1_linear = np.array([y_test_pred_linear_1_s, y_test_pred_linear_1, "ts1 - linear" ])
test_pair_2_linear = np.array([y_test_pred_linear_2_s, y_test_pred_linear_2, "ts2 - linear"])
test_pair_3_linear = np.array([y_test_pred_linear_3_s, y_test_pred_linear_3, "ts3 - linear"])
test_pair_all_linear = np.array([y_test_pred_linear_all_s, y_test_pred_linear_all, "ts - linear"])


train_pair_1_rbf = np.array([y_train_pred_rbf_1_s, y_train_pred_rbf_1, "tr1 - rbf"])
train_pair_2_rbf  = np.array([y_train_pred_rbf_2_s, y_train_pred_rbf_2, "tr2 - rbf"])
train_pair_3_rbf  = np.array([y_train_pred_rbf_3_s, y_train_pred_rbf_3, "tr3 - rbf"])
train_pair_all_rbf  = np.array([y_train_pred_linear_all_s, y_train_pred_linear_all, "tr - rbf"])


# In[230]:



test_pair_1_rbf  = [y_test_pred_rbf_1_s, y_test_pred_rbf_1, "ts1 - rbf" ]
test_pair_2_rbf  = [y_test_pred_rbf_2_s, y_test_pred_rbf_2, "ts2 - rbf" ]
test_pair_3_rbf  = [y_test_pred_rbf_3_s, y_test_pred_rbf_3, "ts3 - rbf"]
test_pair_all_rbf  =[y_test_pred_rbf_all_s, y_test_pred_rbf_all, "ts - rbf" ]


all_pairs = [train_pair_1_linear,train_pair_2_linear, train_pair_3_linear,train_pair_all_linear,
                     test_pair_1_linear, test_pair_2_linear, test_pair_3_linear, test_pair_all_linear,
                     train_pair_1_rbf, train_pair_2_rbf, train_pair_3_rbf, train_pair_all_rbf,
                     test_pair_1_rbf, test_pair_2_rbf, test_pair_3_rbf, test_pair_all_rbf]




# In[231]:


eval_mc_nemar(y_train_pred_linear_1, y_train_pred_rbf_1, "test")


# In[234]:


for pair in all_pairs:
    eval_mc_nemar(pair[0], pair[1], pair[2])


# In[218]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




