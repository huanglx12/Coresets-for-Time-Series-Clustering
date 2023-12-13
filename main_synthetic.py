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
    file_name = './result/N' + str(N) + 'T' + str(T) + 'k' + str(k) + 'lam[' + str(lam_l) + ',' + str(lam_u) + '].csv'
    df.to_csv(file_name)


#####################################
if __name__ == "__main__":
    N = int(sys.argv[1])
    T = [int(sys.argv[2]) for i in range(N)]
    k = int(sys.argv[3])
    times = int(sys.argv[4])
    lam_l = float(sys.argv[5])
    lam_u = float(sys.argv[6])
    d = 2
    time_series = generate_time_series(N, T, d, k, lam_l, lam_u)
    T_i = len(time_series[0])
    file_name = "./result/time_series_data_N" + str(N) + 'T' + str(T_i) + 'k' + str(k) + 'lam[' + str(lam_l) + ',' + str(lam_u) + ".npy"
    np.save(file_name, time_series)

    # compute statistics
    eps, evaluate_data, opt_time, evaluate_coreset_stat, evaluate_uniform_stat, evaluate_LFKF_stat, size_GMM_stat, construction_time_coreset_stat, opt_time_coreset_stat, construction_time_uniform_stat, \
    opt_time_uniform_stat, construction_time_LFKF_stat, opt_time_LFKF_stat = eva.evaluate_GMM(time_series, k, lam_l, lam_u, times)
    arrays_write_to_excel(N, T_i, k, lam_l, lam_u, eps, evaluate_data, opt_time, evaluate_coreset_stat, \
                          evaluate_uniform_stat, evaluate_LFKF_stat, size_GMM_stat, construction_time_coreset_stat, opt_time_coreset_stat, \
                          construction_time_uniform_stat, opt_time_uniform_stat, construction_time_LFKF_stat, opt_time_LFKF_stat)


