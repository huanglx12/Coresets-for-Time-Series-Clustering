import numpy as np
from sklearn.cluster import KMeans
import scipy.special

# sum of squares in a list
def square_sum(list):
    sum = 0
    for i in range(len(list)):
        sum += list[i]*list[i]
    return sum

#################################################
# uniform sampling
def uniform(time_series,nt_sample):
    # construct shape
    N = time_series.shape[0]
    T = np.array([len(time_series[i]) for i in range(N)])
    d = len(time_series[0][0])
    size = sum(T)

    id_list = [i for i in range(size)]
    weight = [0 for i in range(size)]
    sample = np.sort(np.random.choice(id_list, size=nt_sample, replace = False))
    for i in range(nt_sample):
        weight[sample[i]] += 1

    # the coreset form
    symbol = [0 for i in range(N)]
    temp_coreset = []
    for j in range(nt_sample):
        i = 0
        t = sample[j]
        while t >= T[i]:
            t = t - T[i]
            i = i + 1
        symbol[i] = 1
        temp_coreset.append([i, t])

    coreset = []
    count = [0 for i in range(N)]
    id = 0
    for i in range(N):
        if symbol[i] == 1:
            coreset.append([i, N/sum(symbol)])
            while id < len(temp_coreset):
                if temp_coreset[id][0] == i:
                    count[i] += 1
                    id += 1
                else:
                    break
    id = 0
    for s in range(len(coreset)):
        i = coreset[s][0]
        while id < len(temp_coreset):
            if temp_coreset[id][0] == i:
                coreset[s].append([temp_coreset[id][1], T[i]/count[i]])
                id += 1
            else:
                break

    return coreset

################################################
################################################
# LFKF [Lucic et al., JMLR2017]
def LFKF(time_series,k, nt_sample):
    # construct shape
    N = time_series.shape[0]
    T = np.array([len(time_series[i]) for i in range(N)])
    d = len(time_series[0][0])
    NT = sum(T)

    pair_list = [i for i in range(NT)]
    weight = [0 for i in range(nt_sample)]

    # LFKF
    sen = sen_LFKF(time_series, k)
    total_sen = sum(sen)
    pr = [sen[i]/total_sen for i in range(NT)]  # sampling distribution
    sample = np.sort(np.random.choice(pair_list, size=int(nt_sample), p=pr, replace = False))
    weight = [total_sen/sen[sample[i]]/nt_sample for i in range(nt_sample)]

    # the coreset form
    symbol = [0 for i in range(N)]
    temp_coreset = []
    for j in range(nt_sample):
        i = 0
        t = sample[j]
        w = weight[j]
        while t >= T[i]:
            t = t - T[i]
            i = i + 1
        symbol[i] = 1
        temp_coreset.append([i, t, w])

    coreset = []
    count = [0 for i in range(N)]
    id = 0
    for i in range(N):
        if symbol[i] == 1:
            coreset.append([i, N/sum(symbol)])
            while id < len(temp_coreset):
                if temp_coreset[id][0] == i:
                    count[i] += 1
                    id += 1
                else:
                    break
    id = 0
    for s in range(len(coreset)):
        i = coreset[s][0]
        while id < len(temp_coreset):
            if temp_coreset[id][0] == i:
                coreset[s].append([temp_coreset[id][1], sum(symbol) * temp_coreset[id][2] / N])
                id += 1
            else:
                break

    return coreset

################################################
# sensitivity of LFKF
def sen_LFKF(time_series,k):
    # construct shape
    N = time_series.shape[0]
    T = np.array([len(time_series[i]) for i in range(N)])
    d = len(time_series[0][0])
    NT = sum(T)

    # sensitivity function
    sen = [0 for i in range(NT)]

    # k-means clustering
    x = time_series.reshape(NT, d)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
    C = kmeans.cluster_centers_
    p = kmeans.predict(x)
    OPT = 0
    for i in range(NT):
        OPT += square_sum(x[i] - C[p[i]])
    num = [0 for i in range(k)]
    for i in range(NT):
        num[p[i]] += 1

    # compute sensitivities
    senc = [1/num[p[i]] for i in range(NT)]
    for i in range(NT):
        sen[i] = 4 * square_sum(x[i] - C[p[i]])/OPT + 3 * senc[i]

    return sen

################################################
################################################
# individual-level sensitivity function for GMM time-series clustering
def sen_individual(time_series,k):
    # construct shape
    N = time_series.shape[0]
    T = np.array([len(time_series[i]) for i in range(N)])
    d = len(time_series[0][0])

    # sensitivity function
    sen = [0 for i in range(N)]

    # k-means clustering
    b = [sum(np.array(time_series[i]))/T[i] for i in range(N)]
    a = [-square_sum(sum(np.array(time_series[i])))/T[i]/T[i] for i in range(N)]
    for i in range(N):
        for t in range(T[i]):
            a[i] += square_sum(time_series[i][t])/T[i]
    kmeans = KMeans(n_clusters = k, random_state = 0).fit(b)
    C = kmeans.cluster_centers_
    p = kmeans.predict(b)
    OPT = 0
    for i in range(N):
        OPT += square_sum(b[i]-C[p[i]])
    num = [0 for i in range(k)]
    for i in range(N):
        num[p[i]] += 1

    # compute sensitivities
    senc = [1/num[p[i]] for i in range(N)]
    for i in range(N):
        sen[i] = 4 * square_sum(b[i] - C[p[i]])/(OPT + sum(a)) + 3 * senc[i]
    
    return sen

################################################
# time-level sensitivity function of a certain individual for time-series clustering
def sen_time(time_series_individual):
    T = len(time_series_individual)

    # 1-means clustering
    c = sum(np.array(time_series_individual))/T
    OPT = 0
    for t in range(T):
        OPT += square_sum(time_series_individual[t]-c)

    # sensitivity function for a series
    sen = [2*square_sum(time_series_individual[t]-c)/OPT + 6/T for t in range(T)]
    for t in range(T-1):
        sen[T-t-1] += sen[T-t-2]

    return sen

############################################################
# coreset construction for GMM time-series clustering
def coreset_GMM(time_series,k, n_sample,t_sample):
    N = time_series.shape[0]
    T = np.array([len(time_series[i]) for i in range(N)])
    d = len(time_series[0][0])
    coreset = []

    # construct I_S
    sen = sen_individual(time_series,k)
    total_sen = sum(sen)
    pr = [0 for i in range(N)]  # sampling distribution
    for i in range(N):
        pr[i] = sen[i]/total_sen
    id_list = [i for i in range(N)]
    weight = [0 for i in range(N)]
    sample = np.sort(np.random.choice(id_list, size=n_sample, p=pr))
    for i in range(n_sample):
        weight[sample[i]] += 1 / (pr[sample[i]] * n_sample)

    for i in range(N):
        if weight[i] > 0:
            coreset.append([i,weight[i]])

    # construct J_{S,i}
    size_i = len(coreset)
    size = 0
    for s in range(size_i):
        i = coreset[s][0]
        T_i = len(time_series[i])
        sen_i = sen_time(time_series[i])
        total_sen_i = sum(sen_i)
        pr = [0 for i in range(T_i)]  # sampling distribution
        for t in range(T_i):
            pr[t] = sen_i[t] / total_sen_i

        time_list = [t for t in range(T_i)]
        weight = [0 for t in range(T_i)]
        # sample = np.sort(np.random.choice(time_list, size = int(total_sen_i * t_sample), p=pr))
        sample = np.sort(np.random.choice(time_list, size=int(t_sample), p=pr))
        for t in range(t_sample):
            # weight[sample[t]] += 1 / (pr[sample[t]] * int(total_sen_i * t_sample))
            weight[sample[t]] += 1 / (pr[sample[t]] * int(t_sample))

        for t in range(T_i):
            if weight[t] > 0:
                size += 1
                coreset[s].append([t, weight[t]])

    return coreset, size



