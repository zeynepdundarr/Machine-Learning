import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

def angular_second_moment(norm_co_occ):
    return sum([pow(norm_co_occ[i][j], 2) for i in range(len(norm_co_occ)) for j in range(len(norm_co_occ[i]))])

def entropy(norm_co_occ):
    return -1 * sum([pow(norm_co_occ[i][j], 2) for i in range(len(norm_co_occ)) for j in range(len(norm_co_occ[i]))])

def inv_diff_mom(norm_co_occ):
    return  sum([norm_co_occ[i][j] / (1 + pow((i - j), 2)) for i in range(len(norm_co_occ)) for j in range(len(norm_co_occ[i]))])

def contrast(norm_co_occ):
    return sum([pow((i-j),2) * norm_co_occ[i][j] for i in range(len(norm_co_occ)) for j in range(len(norm_co_occ[i]))])

def max_prob(norm_co_occ):
    return np.max(norm_co_occ)

def correlation(norm_co_occ):

  # formula on return
    Ni_arr = [sum(norm_co_occ[i][:]) for i in range(len(norm_co_occ))]
    Nj_arr = [sum(norm_co_occ[:][j]) for j in range(len(norm_co_occ[0]))]

    oi = np.std(Ni_arr)
    oj = np.std(Nj_arr)

    mi = np.mean(Ni_arr)
    mj = np.mean(Nj_arr)

    return sum([i * j * norm_co_occ[i][j] - mi * mj for i in range(len(norm_co_occ)) for j in range(len(norm_co_occ[i]))]) / oi * oj # not sure


def calculateCooccurrenceFeatures(norm_co_occ, normalized):
    all_features = [angular_second_moment(norm_co_occ), max_prob(norm_co_occ), inv_diff_mom(norm_co_occ), contrast(norm_co_occ),
              entropy(norm_co_occ), correlation(norm_co_occ)]
    if not normalized:
        return all_features
    else:
       # normed = np.linalg.norm(all_features)/ norm
        normalize(all_features)
       # normed = (all_features - all_features.mean(axis=0)) / all_features.std(axis=0)
    return normed 

def normalize(data):
   
    data = np.arange(16).reshape((4,4))
    std = np.std(data, axis=1)
    mean = np.mean(data, axis=1)
    normalized = (data - mean.T)/std.T
    return normalized

if __name__ == "__main__":
  clf_linear_s = train(X_train_all_s, balanced_label_all, "linear")
  y_train_pred_linear_all, y_train_pred_linear_all = test(X_train_all_s, X_test_all_s, balanced_label_all, test_all_label,
                                                         clf_linear_s, "0i - linear")
  y_train_pred_linear_1, y_train_pred_linear_1 = test(X_train_1_s, X_test_1_s, balanced_label_1, test_1_label, clf_linear_s,
                                                     "1b - linear")  # add x train1 and y_train1
  y_train_pred_linear_2, y_train_pred_linear_2 = test(X_train_2_s, X_test_2_s, balanced_label_2, test_2_label, clf_linear_s,
                                                     "2b - linear")
  y_train_pred_linear_3, y_train_pred_linear_3 = test(X_train_3_s, X_test_3_s, balanced_label_3, test_3_label, clf_linear_s,
                                                     "3b - linear")

  clf_rbf_s = train(X_train_all_s, balanced_all_label, "rbf")
  y_train_pred_rbf_all, y_train_pred_rbf_all = test(X_train_all_s, X_test_all_s, balanced_all_label, test_all_label, clf_rbf_s,
                                                   "0b - rbf ")
  y_train_pred_rbf_1, y_train_pred_rbf_1 = test(X_train_1_s, X_test_1_s, balanced_label_1, test_1_label, clf_rbf_s,
                                               "1b - rbf")
  y_train_pred_rbf_2, y_train_pred_rbf_2 = test(X_train_2_s, X_test_2_s, balanced_label_2, test_2_label, clf_rbf_s,
                                               "2b - rbf")
  y_train_pred_rbf_3, y_train_pred_rbf_3 = test(X_train_3_s, X_test_3_s, balanced_label_3, test_3_label, clf_rbf_s,
                                               "3b - rbf")

  train_pair_1 = np.array([y_train_pred_linear_1, y_train_pred_rbf_1, "1tr"])
  train_pair_2 = np.array([y_train_pred_linear_2, y_train_pred_rbf_2, "2tr"])
  train_pair_3 = np.array([y_train_pred_linear_3, y_train_pred_rbf_3, "3tr"])
  train_pair_all = np.array([y_train_pred_linear_all, y_train_pred_rbf_all, "0tr"])

  test_pair_1 = np.array([y_train_pred_linear_1, y_train_pred_rbf_1, "1tr"])
  test_pair_2 = np.array([y_train_pred_linear_2, y_train_pred_rbf_2, "2tr"])
  test_pair_3 = np.array([y_train_pred_linear_3, y_train_pred_rbf_3, "3tr"])
  test_pair_all = np.array([y_train_pred_linear_all, y_train_pred_rbf_all, "0tr"])

  all_pairs = np.array([train_pair_1, train_pair_2, train_pair_3, train_pair_all
                         , test_pair_1, test_pair_2, test_pair_3, test_pair_all])

  #
  eval_mc_nemar(y_train_pred_linear_1, y_train_pred_rbf_1)

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