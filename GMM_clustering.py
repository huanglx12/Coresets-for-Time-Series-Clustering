import numpy as np
import coreset as co

#####################################
# the objective function of GMM on the full dataset w.r.t. distribution alpha, Gaussian means mu, Gaussian covariance precision, and autocorrelation Lambda
def GMM_obj(time_series, alpha, mu, precision, Lambda):
    # construct shape
    N = time_series.shape[0]
    T = np.array([len(time_series[i]) for i in range(N)])
    d = len(time_series[0][0])
    k = len(alpha)
    size = sum(T)

    # compute GMM clustering objective
    detprecision = [np.linalg.det(precision[l]) for l in range(k)] # determinant of covariance matrix
    obj = 0
    for i in range(N):
        obj_i = 0
        temp = [0 for l in range(k)]
        for l in range(k):
            V = [np.array(time_series[i][t] - mu[l]) for t in range(T[i])]
            temp_obj = V[0].dot(precision[l].dot(V[0]))
            for t in range(T[i] - 1):
                v = np.array(V[t + 1] - Lambda[l].dot(V[t]))
                temp_obj += v.dot(precision[l].dot(v))
            temp[l] = temp_obj
        min_i = np.min(temp)
        for l in range(k):
            obj_i += np.where(alpha[l] >= 0, alpha[l], 0) * np.exp(- (temp[l] - min_i) / 2 / T[i]) * np.sqrt(
                detprecision[l]) / np.sqrt(
                np.power(2 * np.pi, d / 2))
        obj -= np.log(obj_i) - min_i / 2 / T[i]
    return obj

#####################################
# the objective function of glse on the coreset
def GMM_coreset_obj(time_series, coreset, alpha, mu, precision, Lambda):
    # construct shape
    N = time_series.shape[0]
    T = np.array([len(time_series[i]) for i in range(N)])
    d = len(time_series[0][0])
    k = len(alpha)

    # compute GMM clustering objective
    detprecision = [np.linalg.det(precision[l]) for l in range(k)]  # determinant of covariance matrix
    obj = 0
    for s in range(len(coreset)):
        i = coreset[s][0]
        obj_i = 0
        temp = [0 for l in range(k)]
        for l in range(k):
            V = [np.array(time_series[i][t] - mu[l]) for t in range(T[i])]
            temp_obj = 0
            for t in range(len(coreset[s]) - 2):
                if int(coreset[s][t + 2][0]) == 0:
                    v = V[0]
                else:
                    v = np.array(V[coreset[s][t + 2][0]] - Lambda[l].dot(V[coreset[s][t + 2][0] - 1]))
                temp_obj += v.dot(precision[l].dot(v)) * coreset[s][t + 2][1]
            temp[l] = temp_obj
        min_i = np.min(temp)
        for l in range(k):
            obj_i += np.where(alpha[l] >= 0, alpha[l], 0) * np.exp(- (temp[l] - min_i) / 2 / T[i]) * np.sqrt(
                detprecision[l]) / np.sqrt(
                np.power(2 * np.pi, d / 2))
        obj -= (np.log(obj_i) - min_i / 2 / T[i]) * coreset[s][1]
    return obj

##############################################
