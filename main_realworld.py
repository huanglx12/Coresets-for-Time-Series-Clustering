from GMM_clustering import *
from coreset import *
from generate import *
import pandas as pd
import time
import evaluation as eva
import pickle
import sys

##########################################
# output an excel that records statistics
def arrays_write_to_excel(N, T, k, lam_l, lam_u, eps, evaluate_data, opt_time, evaluate_coreset_stat, evaluate_uniform_stat, evaluate_LFKF_stat, size_GMM_stat, construction_time_coreset_stat, opt_time_coreset_stat, construction_time_uniform_stat, \
           opt_time_uniform_stat, construction_time_LFKF_stat, opt_time_LFKF_stat):
    l = []
    for e in range(len(eps)):
        l.append([str(eps[e]), \
                  str(round(evaluate_coreset_stat[e][0], 2)) + ' (' + str(round(evaluate_coreset_stat[e][1], 2)) + ')', \
                  str(round(evaluate_uniform_stat[e][0], 2)) + ' (' + str(round(evaluate_uniform_stat[e][1], 2)) + ')', \
                  str(round(evaluate_LFKF_stat[e][0], 2)) + ' (' + str(round(evaluate_LFKF_stat[e][1], 2)) + ')', \
                  str(round(evaluate_data, 2)), \
                  str(round(size_GMM_stat[e][0], 2)) + ' (' + str(round(size_GMM_stat[e][1],2)) + ')', \
                  str(round(construction_time_coreset_stat[e][0], 2)) + ' (' + str(round(construction_time_coreset_stat[e][1], 2)) + ')', \
                  str(round(opt_time_coreset_stat[e][0], 2)) + ' (' + str(round(opt_time_coreset_stat[e][1], 2)) + ')', \
                  str(round(construction_time_uniform_stat[e][0], 2)) + ' (' + str(round(construction_time_uniform_stat[e][1], 2)) + ')', \
                  str(round(opt_time_uniform_stat[e][0], 2)) + ' (' + str(round(opt_time_uniform_stat[e][1], 2)) + ')', \
                  str(round(construction_time_LFKF_stat[e][0], 2)) + ' (' + str(round(construction_time_LFKF_stat[e][1], 2)) + ')', \
                  str(round(opt_time_LFKF_stat[e][0], 2)) + ' (' + str(round(opt_time_LFKF_stat[e][1], 2)) + ')', \
                  str(round(opt_time, 2))])

    arr = np.asarray(l)
    df = pd.DataFrame(arr, columns=['eps', 'value_c', 'value_u', 'value_LFKF', 'value full', 'size', 'T_C', 'T_C+T_S', 'T_U', \
                                    'T_U + T_S', 'T_LFKF', 'T_LFKF + T_S', 'T_X'])
    file_name = './result/Yale_Acc_N' + str(N) + 'T' + str(T) + 'k' + str(k) + 'lam[' + str(lam_l) + ',' + str(lam_u) + '].csv'
    df.to_csv(file_name)


#####################################
if __name__ == "__main__":
    k = int(sys.argv[1])
    times = int(sys.argv[2])
    lam_l = float(sys.argv[3])
    lam_u = float(sys.argv[4])
    num = int(sys.argv[5])
    panel = np.load('realworld.npy')
    panel = panel[1000:1000+num,:,[4,8]]
    min_0 = np.array([np.min(panel[i, :, 0]) for i in range(len(panel))])
    max_0 = np.array([np.max(panel[i, :, 0]) for i in range(len(panel))])
    min_1 = np.array([np.min(panel[i, :, 1]) for i in range(len(panel))])
    max_1 = np.array([np.max(panel[i, :, 1]) for i in range(len(panel))])
    min0 = np.min(min_0)
    max0 = np.max(max_0)
    min1 = np.min(min_1)
    max1 = np.max(max_1)
    time_series = np.array([[[100*(panel[i][j][0] - min0)/(max0-min0), 100*(panel[i][j][1] - min1)/(max1-min1)] for j in range(len(panel[0]))] for i in range(len(panel))])
    N = len(time_series)
    T = len(time_series[0])
    d = len(time_series[0][0])
    T_i = len(time_series[0])

    # compute statistics
    eps, evaluate_data, opt_time, evaluate_coreset_stat, evaluate_uniform_stat, evaluate_LFKF_stat, size_GMM_stat, construction_time_coreset_stat, opt_time_coreset_stat, construction_time_uniform_stat, \
    opt_time_uniform_stat, construction_time_LFKF_stat, opt_time_LFKF_stat = eva.evaluate_GMM(time_series, k, lam_l, lam_u, times)
    arrays_write_to_excel(N, T_i, k, lam_l, lam_u, eps, evaluate_data, opt_time, evaluate_coreset_stat, \
                          evaluate_uniform_stat, evaluate_LFKF_stat, size_GMM_stat, construction_time_coreset_stat, opt_time_coreset_stat, \
                          construction_time_uniform_stat, opt_time_uniform_stat, construction_time_LFKF_stat, opt_time_LFKF_stat)


