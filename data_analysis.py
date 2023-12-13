from GMM_clustering import *
from coreset import *
from generate import *
import pandas as pd
import time
import evaluation as eva
import pickle
import sys


#############################################
# record empirical errors by an excel for drawing boxplot
def arrays_to_boxplot(eps, GMM_0, uniform_0, GMM_1, uniform_1):
    l = []
    T = len(GMM_0[0])
    for i in range(len(eps)):
        for t in range(T):
            l.append([eps[i], GMM_0[i][t], uniform_0[i][t], GMM_1[i][t], uniform_1[i][t]])
    arr = np.asarray(l)
    df = pd.DataFrame(arr, columns = ['eps', 'CGLSE (Gaussian)', 'Uni (Gaussian)', 'CGLSE (Cauchy)', 'Uni (Cauchy)'])
    df.to_csv('emp' + '_synthetic.csv')
    return


#############################################
