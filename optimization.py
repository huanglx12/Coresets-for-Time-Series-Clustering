import numpy as np
from GMM_clustering import *
from scipy.optimize import minimize, LinearConstraint
from sklearn.mixture import GaussianMixture
from generate import *
from GMM_clustering import *
from coreset import *
import sys

########################################################
# GMM
########################################################
# optimization for GMM time-series clustering on the full dataset


def opt_GMM(time_series, k, lam_l, lam_u):
    N = time_series.shape[0]
    T = np.array([len(time_series[i]) for i in range(N)])
    d = len(time_series[0][0])

    # optimization for GMM time-series clustering on the full dataset
    def GMM(alpha, mu, precision_cho, diagLambda):
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]

        # compute GMM clustering objective
        precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
        detprecision = [np.linalg.det(precision[l]) for l in range(k)]  # determinant of covariance matrix
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
                obj_i += np.where(alpha[l] >= 0, alpha[l], 0) * np.exp(- (temp[l] - min_i) / 2 / T[i]) * np.sqrt(detprecision[l]) / np.sqrt(
                    np.power(2 * np.pi, d / 2))
            temp = - np.log(obj_i) + min_i / 2 / T[i]
            # print(temp)
            obj += temp
        return obj

    # computing individual probabilities under fixed model parameters
    def GMM_individual_prob(alpha, mu, precision_cho, diagLambda):
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]

        # compute GMM clustering objective for each individual
        precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
        detprecision = [np.linalg.det(precision[l]) for l in range(k)]  # determinant of covariance matrix
        obj = [[0 for l in range(k)] for i in range(N)]
        temp = [[0 for l in range(k)] for i in range(N)]
        for i in range(N):
            for l in range(k):
                V = [np.array(time_series[i][t] - mu[l]) for t in range(T[i])]
                temp_obj = V[0].dot(precision[l].dot(V[0]))
                for t in range(T[i] - 1):
                    v = np.array(V[t + 1] - Lambda[l].dot(V[t]))
                    temp_obj += v.dot(precision[l].dot(v))
                temp[i][l] = temp_obj

        for i in range(N):
            min_i = np.min(temp[i])
            for l in range(k):
                obj[i][l] = np.where(alpha[l] >= 0, alpha[l], 0) * np.exp(- (temp[i][l] - min_i) / 2 / T[i]) * np.sqrt(detprecision[l])
        prob = [[0 for l in range(k)] for i in range(N)]
        for i in range(N):
            total_prob_i = sum(obj[i])
            for l in range(k):
                prob[i][l] = obj[i][l] / total_prob_i
        return prob

    # compute alpha from prob
    def GMM_alpha(prob):
        N = len(prob)
        alpha = [0 for l in range(k)]
        for i in range(N):
            for l in range(k):
                alpha[l] += prob[i][l] / N
        return alpha

    # optimizing mu for GMM time-series clustering on the full dataset
    def GMM_mu(x, *args):
        mu = x.reshape((k,d))
        prob = args[0]
        precision_cho = args[1]
        diagLambda = args[2]
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]

        # compute GMM clustering objective
        precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
        detprecision = [np.linalg.det(precision[l]) for l in range(k)]  # determinant of covariance matrix
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
                obj_i += prob[i][l] * np.exp(- (temp[l] - min_i) / 2 / T[i]) * np.sqrt(
                    detprecision[l]) / np.sqrt(
                    np.power(2 * np.pi, d / 2))
            obj -= np.log(obj_i) - min_i / 2 / T[i]
        return obj

    # optimizing precision_cho for GMM time-series clustering on the full dataset
    def GMM_precision_cho(x, *args):
        A = x.reshape((k, int(d*(d+1)/2)))
        precision_cho = [[[0 for i in range(d)] for j in range(d)] for l in range(k)]
        for l in range(k):
            for s1 in range(d):
                for s2 in range(s1+1):
                    precision_cho[l][s1][s2] = A[l][int(s1*(s1+1)/2 + s2)]
        prob = args[0]
        mu = args[1]
        diagLambda = args[2]
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]

        # compute GMM clustering objective
        precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
        detprecision = [np.linalg.det(precision[l]) for l in range(k)]  # determinant of covariance matrix
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
                obj_i += prob[i][l] * np.exp(- (temp[l] - min_i) / 2 / T[i]) * np.sqrt(
                    detprecision[l]) / np.sqrt(
                    np.power(2 * np.pi, d / 2))
            obj -= np.log(obj_i) - min_i / 2 / T[i]
        return obj

    # optimizing diagLambda for GMM time-series clustering on the full dataset
    def GMM_diagLambda(x, *args):
        diagLambda = x.reshape((k,d))
        prob = args[0]
        mu = args[1]
        precision_cho = args[2]
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]

        # compute GMM clustering objective
        precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
        detprecision = [np.linalg.det(precision[l]) for l in range(k)]  # determinant of covariance matrix
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
                obj_i += prob[i][l] * np.exp(- (temp[l] - min_i) / 2 / T[i]) * np.sqrt(
                    detprecision[l]) / np.sqrt(
                    np.power(2 * np.pi, d / 2))
            obj -= np.log(obj_i) - min_i / 2 / T[i]
        return obj

    # initialize diagLambda
    def init_diagLambda(alpha, mu, precision_cho):
        flag = 0
        while flag == 0:
            diagLambda = [np.array(generate_diagLambda(d, lam_l, lam_u)) for l in range(k)]
            value = GMM(alpha, mu, precision_cho, diagLambda)
            if value <= 1e+4:
                flag = 1
        return diagLambda

    # optimization
    def slsqp(alpha, mu, precision_cho, diagLambda, init_value):

        # constraints for diagLambda
        # Ineq. constraints
        coeff = [[0 for i in range(k*d)] for j in range(k * d)]
        for i in range(k * d):
            coeff[i][i] = 1
        lb = [lam_l for i in range(k * d)]
        ub = [lam_u for i in range(k * d)]
        linear_constraint_diagLambda = LinearConstraint(coeff, lb, ub)

        # iteratively optimize
        flag = 0
        value = init_value
        # min = value
        while flag == 0:
            prob = GMM_individual_prob(alpha, mu, precision_cho, diagLambda)
            x0 = np.array(diagLambda).flatten()
            res = minimize(GMM_diagLambda, x0, args=(prob, mu, precision_cho), method='SLSQP', constraints=[linear_constraint_diagLambda],
                           options={'ftol': 1e-2, 'eps': 1e-2, 'disp': False, 'maxiter': 100})
            diagLambda = res.x.reshape((k, d))

            x0 = np.array(mu).flatten()
            res = minimize(GMM_mu, x0, args=(prob, precision_cho, diagLambda), method='SLSQP', options={'ftol': 1e-2, 'eps': 1e-2, 'disp': False, 'maxiter': 100})
            mu = res.x.reshape((k,d))

            temp_precision_cho = [[0 for i in range(int(d * (d + 1) / 2))] for l in range(k)]
            for l in range(k):
                for s1 in range(d):
                    for s2 in range(s1 + 1):
                        temp_precision_cho[l][int(s1 * (s1 + 1) / 2) + s2] = precision_cho[l][s1][s2]
            x0 = np.array(temp_precision_cho).flatten()
            res = minimize(GMM_precision_cho, x0, args=(prob, mu, diagLambda), method='SLSQP', options={'ftol': 1e-2, 'eps': 1e-2, 'disp': False, 'maxiter': 100})
            A = res.x.reshape((k, int(d * (d + 1) / 2)))
            precision_cho = [[[0 for i in range(d)] for j in range(d)] for l in range(k)]
            for l in range(k):
                for s1 in range(d):
                    for s2 in range(s1 + 1):
                        precision_cho[l][s1][s2] = A[l][int(s1 * (s1 + 1) / 2 + s2)]

            alpha = GMM_alpha(prob)
            temp_value = GMM(alpha, mu, precision_cho, diagLambda)
            # min = np.min([min, temp_value])
            if temp_value >= value + 1e+1:
                return []
            if temp_value >= value - 1e-2:
                flag = 1
            value = temp_value

        model = np.array([alpha, mu, precision_cho, diagLambda, value])
        return model


    # Initialization
    X = [time_series[i][0] for i in range(N)]
    gm = GaussianMixture(n_components=k, random_state=0, covariance_type="full").fit(X)
    alpha = gm.weights_
    mu = gm.means_
    Cov = gm.covariances_.reshape((k, d, d))
    precision = [np.linalg.inv(Cov[l]) for l in range(k)]
    precision_cho = [np.linalg.cholesky(precision[l]) for l in range(k)]
    # precision_cho = [np.random.rand(d,d) for l in range(k)]
    # precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
    # diagLambda = [[0 for s in range(d)] for l in range(k)]
    # Lambda = [np.diag(diagLambda[l]) for l in range(k)]
    # value = GMM(alpha, mu, precision_cho, diagLambda)

    flag = 0
    while flag == 0:
        diagLambda = init_diagLambda(alpha, mu, precision_cho)
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]
        init_value = GMM(alpha, mu, precision_cho, diagLambda)
        model = slsqp(alpha, mu, precision_cho, diagLambda, init_value)
        if len(model) > 0:
            print("Initialized_alpha:", alpha)
            print("Initialized_mu:", mu)
            print("Initialized_precision:", precision)
            print("Initialized_Lambda:", Lambda)
            print("Initialized_value:", init_value)
            alpha = model[0]
            mu = model[1]
            precision_cho = model[2]
            diagLambda = model[3]
            value = model[4]
            flag = 1

    precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
    Lambda = [np.diag(diagLambda[l]) for l in range(k)]

    print("Optimized_alpha:", alpha)
    print("Optimized_mu:", mu)
    print("Optimized_precision:", precision)
    print("Optimized_Lambda:", Lambda)
    print("Optimized_value:", GMM_obj(time_series, alpha, mu, precision, Lambda))
    return value

#####################################
# optimization for GMM time-series clustering on the coreset

def opt_GMM_coreset(time_series, coreset, k, lam_l, lam_u):
    N = time_series.shape[0]
    T = np.array([len(time_series[i]) for i in range(N)])
    d = len(time_series[0][0])

    # minimize for beta
    def GMM_coreset(alpha, mu, precision_cho, diagLambda):
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]

        # compute GMM clustering objective
        precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
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
                    if coreset[s][t + 2][0] == 0:
                        v = V[0]
                    else:
                        v = np.array(V[coreset[s][t + 2][0]] - Lambda[l].dot(V[coreset[s][t + 2][0] - 1]))
                    temp_obj += v.dot(precision[l].dot(v) * coreset[s][t + 2][1])
                temp[l] = temp_obj
            min_i = np.min(temp)
            for l in range(k):
                obj_i += np.where(alpha[l] >= 0, alpha[l], 0) * np.exp(- (temp[l] - min_i) / 2 / T[i]) * np.sqrt(detprecision[l]) / np.sqrt(
                    np.power(2 * np.pi, d / 2))
            obj -= (np.log(obj_i) - min_i / 2 / T[i])* coreset[s][1]
        return obj

    # computing individual probabilities under fixed model parameters
    def GMM_coreset_individual_prob(alpha, mu, precision_cho, diagLambda):
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]

        # compute GMM clustering objective for each individual
        precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
        detprecision = [np.linalg.det(precision[l]) for l in range(k)]  # determinant of covariance matrix
        obj = [[0 for l in range(k)] for s in range(len(coreset))]
        temp = [[0 for l in range(k)] for s in range(len(coreset))]
        for s in range(len(coreset)):
            i = coreset[s][0]
            for l in range(k):
                V = [np.array(time_series[i][t] - mu[l]) for t in range(T[i])]
                temp_obj = 0
                for t in range(len(coreset[s]) - 2):
                    if coreset[s][t + 2][0] == 0:
                        v = V[0]
                    else:
                        v = np.array(V[coreset[s][t + 2][0]] - Lambda[l].dot(V[coreset[s][t + 2][0] - 1]))
                    temp_obj += v.dot(precision[l].dot(v) * coreset[s][t + 2][1])
                temp[s][l] = temp_obj

        for s in range(len(coreset)):
            i = coreset[s][0]
            min_s = np.min(temp[s])
            for l in range(k):
                obj[s][l] = np.where(alpha[l] >= 0, alpha[l], 0) * np.exp(- (temp[s][l] - min_s) / 2 / T[i]) * np.sqrt(detprecision[l])
        prob = [[0 for l in range(k)] for s in range(len(coreset))]
        for s in range(len(coreset)):
            total_prob_s = sum(obj[s])
            for l in range(k):
                prob[s][l] = obj[s][l] / total_prob_s
        return prob

    # compute alpha from prob
    def GMM_coreset_alpha(prob):
        N_coreset = len(prob)
        alpha = [0 for l in range(k)]
        for s in range(N_coreset):
            for l in range(k):
                alpha[l] += prob[s][l] / N_coreset
        return alpha

    # optimizing mu for GMM time-series clustering on the coreset
    def GMM_coreset_mu(x, *args):
        mu = x.reshape((k,d))
        prob = args[0]
        precision_cho = args[1]
        diagLambda = args[2]
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]

        # compute GMM clustering objective
        precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
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
                    if coreset[s][t + 2][0] == 0:
                        v = V[0]
                    else:
                        v = np.array(V[coreset[s][t + 2][0]] - Lambda[l].dot(V[coreset[s][t + 2][0] - 1]))
                    temp_obj += v.dot(precision[l].dot(v) * coreset[s][t + 2][1])
                temp[l] = temp_obj
            min_i = np.min(temp)
            for l in range(k):
                obj_i += prob[s][l] * np.exp(- (temp[l]- min_i) / 2 / T[i]) * np.sqrt(
                    detprecision[l]) / np.sqrt(
                    np.power(2 * np.pi, d / 2))
            obj -= (np.log(obj_i) - min_i / 2 / T[i]) * coreset[s][1]
        return obj

    # optimizing precision_cho for GMM time-series clustering on the coreset
    def GMM_coreset_precision_cho(x, *args):
        A = x.reshape((k, int(d * (d + 1) / 2)))
        precision_cho = [[[0 for i in range(d)] for j in range(d)] for l in range(k)]
        for l in range(k):
            for s1 in range(d):
                for s2 in range(s1 + 1):
                    precision_cho[l][s1][s2] = A[l][int(s1 * (s1 + 1) / 2 + s2)]
        prob = args[0]
        mu = args[1]
        diagLambda = args[2]
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]

        # compute GMM clustering objective
        precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
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
                    if coreset[s][t + 2][0] == 0:
                        v = V[0]
                    else:
                        v = np.array(V[coreset[s][t + 2][0]] - Lambda[l].dot(V[coreset[s][t + 2][0] - 1]))
                    temp_obj += v.dot(precision[l].dot(v) * coreset[s][t + 2][1])
                temp[l] = temp_obj
            min_i = np.min(temp)
            for l in range(k):
                obj_i += prob[s][l] * np.exp(- (temp[l]- min_i) / 2 / T[i]) * np.sqrt(
                    detprecision[l]) / np.sqrt(
                    np.power(2 * np.pi, d / 2))
            obj -= (np.log(obj_i) - min_i / 2 / T[i]) * coreset[s][1]
        return obj

    # optimizing diagLambda for GMM time-series clustering on the coreset
    def GMM_coreset_diagLambda(x, *args):
        diagLambda = x.reshape((k,d))
        prob = args[0]
        mu = args[1]
        precision_cho = args[2]
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]

        # compute GMM clustering objective
        precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
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
                    if coreset[s][t + 2][0] == 0:
                        v = V[0]
                    else:
                        v = np.array(V[coreset[s][t + 2][0]] - Lambda[l].dot(V[coreset[s][t + 2][0] - 1]))
                    temp_obj += v.dot(precision[l].dot(v) * coreset[s][t + 2][1])
                temp[l] = temp_obj
            min_i = np.min(temp)
            for l in range(k):
                obj_i += prob[s][l] * np.exp(- (temp[l] - min_i) / 2 / T[i]) * np.sqrt(
                    detprecision[l]) / np.sqrt(np.power(2 * np.pi, d / 2))
            obj -= (np.log(obj_i) - min_i / 2 / T[i]) * coreset[s][1]
        return obj

    # initialize diagLambda
    def init_diagLambda(alpha, mu, precision_cho):
        flag = 0
        while flag == 0:
            diagLambda = [np.array(generate_diagLambda(d, lam_l, lam_u)) for l in range(k)]
            value = GMM_coreset(alpha, mu, precision_cho, diagLambda)
            if value <= 1e+4:
                flag = 1
        return diagLambda

    # optimization
    def slsqp_coreset(alpha, mu, precision_cho, diagLambda, init_value):
        # constraints for diagLambda
        # Ineq. constraints
        coeff = [[0 for i in range(k * d)] for j in range(k * d)]
        for i in range(k * d):
            coeff[i][i] = 1
        lb = [lam_l for i in range(k * d)]
        ub = [lam_u for i in range(k * d)]
        linear_constraint_diagLambda = LinearConstraint(coeff, lb, ub)

        # iteratively optimize
        flag = 0
        value = init_value
        # min = value
        while flag == 0:
            prob = GMM_coreset_individual_prob(alpha, mu, precision_cho, diagLambda)
            x0 = np.array(diagLambda).flatten()
            res = minimize(GMM_coreset_diagLambda, x0, args=(prob, mu, precision_cho), method='SLSQP',
                           constraints=[linear_constraint_diagLambda],
                           options={'ftol': 1e-2, 'eps': 1e-2, 'disp': False, 'maxiter': 100})
            diagLambda = res.x.reshape((k, d))

            x0 = np.array(mu).flatten()
            res = minimize(GMM_coreset_mu, x0, args=(prob, precision_cho, diagLambda), method='SLSQP',
                           options={'ftol': 1e-2, 'eps': 1e-2, 'disp': False, 'maxiter': 100})
            mu = res.x.reshape((k, d))

            temp_precision_cho = [[0 for i in range(int(d * (d + 1) / 2))] for l in range(k)]
            for l in range(k):
                for s1 in range(d):
                    for s2 in range(s1 + 1):
                        temp_precision_cho[l][int(s1 * (s1 + 1) / 2) + s2] = precision_cho[l][s1][s2]
            x0 = np.array(temp_precision_cho).flatten()
            res = minimize(GMM_coreset_precision_cho, x0, args=(prob, mu, diagLambda), method='SLSQP',
                           options={'ftol': 1e-2, 'eps': 1e-2, 'disp': False, 'maxiter': 100})
            A = res.x.reshape((k, int(d * (d + 1) / 2)))
            precision_cho = [[[0 for i in range(d)] for j in range(d)] for l in range(k)]
            for l in range(k):
                for s1 in range(d):
                    for s2 in range(s1 + 1):
                        precision_cho[l][s1][s2] = A[l][int(s1 * (s1 + 1) / 2 + s2)]

            alpha = GMM_coreset_alpha(prob)
            temp_value = GMM_coreset(alpha, mu, precision_cho, diagLambda)
            # min = np.min([min, temp_value])
            if temp_value >= value + 1e+1:
                return []
            if temp_value >= value - 1e-2:
                flag = 1
            value = temp_value

        model = np.array([alpha, mu, precision_cho, diagLambda, value])
        return model

    # Initialization
    X = [time_series[i][0] for i in range(N)]
    gm = GaussianMixture(n_components=k, random_state=0, covariance_type="full").fit(X)
    alpha = gm.weights_
    mu = gm.means_
    Cov = gm.covariances_.reshape((k, d, d))
    precision = [np.linalg.inv(Cov[l]) for l in range(k)]
    precision_cho = [np.linalg.cholesky(precision[l]) for l in range(k)]
    # precision_cho = [np.random.rand(d,d) for l in range(k)]
    # precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
    # diagLambda = [[0 for s in range(d)] for l in range(k)]
    # Lambda = [np.diag(diagLambda[l]) for l in range(k)]
    # value = GMM(alpha, mu, precision_cho, diagLambda)

    flag = 0
    while flag == 0:
        diagLambda = init_diagLambda(alpha, mu, precision_cho)
        Lambda = [np.diag(diagLambda[l]) for l in range(k)]
        init_value = GMM_coreset(alpha, mu, precision_cho, diagLambda)
        model = slsqp_coreset(alpha, mu, precision_cho, diagLambda, init_value)
        if len(model) > 0:
            print("Initialized_alpha:", alpha)
            print("Initialized_mu:", mu)
            print("Initialized_precision:", precision)
            print("Initialized_Lambda:", Lambda)
            print("Initialized_value:", init_value)
            alpha = model[0]
            mu = model[1]
            precision_cho = model[2]
            diagLambda = model[3]
            value = model[4]
            flag = 1

    precision = [np.array(precision_cho[l]).dot(np.array(precision_cho[l]).T) for l in range(k)]
    Lambda = [np.diag(diagLambda[l]) for l in range(k)]
    value_full = GMM_obj(time_series, alpha, mu, precision, Lambda)

    print("Optimized_coreset_alpha:", alpha)
    print("Optimized_coreset_mu:", mu)
    print("Optimized_coreset_precision:", precision)
    print("Optimized_coreset_Lambda:", Lambda)
    print("Optimized_coreset_value:", value)
    print("Optimized_coreset_value_full:", value_full)
    return value_full


#######################################################################
# if __name__ == "__main__":
#     N = int(sys.argv[1])
#     T = [int(sys.argv[2]) for i in range(N)]
#     k = 4
#     d = 2
#     lam = 0.1
#     time_series = generate_time_series(N, T, d, k, lam)
#     print("------------------- The full dataset ----------------------")
#     value = opt_GMM(time_series, k, lam)
#
#     # time_series = generate_time_series(N, T, d, k, lam)
#     n_sample = int(sys.argv[3])
#     t_sample = int(sys.argv[4])
#     coreset, size = coreset_GMM(time_series, k, n_sample, t_sample)
#     print("------------------- Coreset ----------------------")
#     print("Size:", size)
#     value_c = opt_GMM_coreset(time_series, coreset, k, lam)
#
#     uniform = uniform(time_series, size)
#     print("------------------- Uniform ----------------------")
#     print("Size:", size)
#     value_u = opt_GMM_coreset(time_series, uniform, k, lam)

