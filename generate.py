import random
import numpy as np
import math
from scipy.stats import cauchy, random_correlation
from GMM_clustering import *
from coreset import *
import time
import pickle
from GMM_clustering import *

# sum of squares in a list
def square_sum(list):
    sum = 0
    for i in range(len(list)):
        sum += list[i]*list[i]
    return sum

##############################################
# generate time_series data
def generate_time_series(N,T,d, k, lam_l, lam_u):
    # parameter generation
    alpha = generate_alpha(k)
    mu = []
    precision = []
    Lambda = []
    for l in range(k):
        mu.append(generate_mu(d))
        precision.append(generate_precision(d))
        Lambda.append(np.diag(generate_diagLambda(d,lam_l,lam_u)))

    id_list = [l for l in range(k)]
    cluster = np.random.choice(id_list, size=N, p = alpha)
    Sigma = [np.linalg.inv(precision[l]) for l in range(k)]

    # generate time-series data
    time_series = [[[0 for s in range(d)] for t in range(T[i])] for i in range(N)]
    for i in range(N):
        error_i = []
        error_basic = np.random.multivariate_normal([0 for s in range(d)], Sigma[cluster[i]], T[i])
        for t in range(T[i]):
            error_i.append(error_basic[t])
            for tt in range(min(t - 1, 1)):
                error_i[t] += np.array(Lambda[cluster[i]]).dot(error_i[t - tt - 1])
        for t in range(T[i]):
            time_series[i][t] = mu[cluster[i]] + error_i[t]

    time_series = np.array(time_series)
    print("Generated_alpha:", alpha)
    print("Generated_mu:", mu)
    print("Generated_precision:", precision)
    print("Generated_Lambda:", Lambda)
    print("Generated_value:", GMM_obj(time_series, alpha, mu, precision, Lambda))
    return time_series

##############################################
# generate alpha
def generate_alpha(k):
    alpha = np.abs(np.random.normal(0, 1, k))
    total_alpha = sum(alpha)
    for i in range(k):
        alpha[i] = alpha[i] / total_alpha
    return alpha

##############################################
# generate mu
def generate_mu(d):
    mu = np.random.normal(0,1,d)
    return mu


##############################################
# generate precision matrix
def generate_precision(d):
    A = np.random.rand(d,d)
    precision = A.dot(A.T)
    return precision

##############################################
# generate Lambda (diagonal)
def generate_diagLambda(d,lam_l,lam_u):
    diagLambda = np.array([random.uniform(lam_l,lam_u) for i in range(d)])
    return diagLambda


##############################################
# time_series = generate_time_series(2,[2,2,3],2, 2, 0.9)
# alpha = np.array([0.5, 0.5])
# mu = np.array([[0,0], [1,1]])
# precision = np.array([[[1,0],[0,1]],[[0.5, 0.5],[0.4,0.6]]])
# Lambda = np.array([[[0.1,0],[0,0.1]],[[0.4,0],[0,0.4]]])
#
# print(GMM_obj(time_series, alpha, mu, precision, Lambda))
#
# coreset, size = coreset_GMM(time_series,2,2,1)
# print(coreset)
# print(GMM_coreset_obj(time_series,coreset, alpha,mu,precision,Lambda))
#
#
# coreset= uniform(time_series,2)
# print(coreset)
# print(GMM_coreset_obj(time_series,coreset, alpha,mu,precision,Lambda))