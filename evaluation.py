from optimization import *

#############################################
# evaluate the empirical error for coreset_GMM and uniform
def evaluate_GMM(time_series,k,lam_l,lam_u,times):
    N = time_series.shape[0]
    T = np.array([len(time_series[i]) for i in range(N)])
    d = len(time_series[0][0])
    eps = [0.5, 0.4, 0.3, 0.2, 0.1]

    evaluate_coreset = [[0 for t in range(times)] for e in range(len(eps))]
    evaluate_uniform = [[0 for t in range(times)] for e in range(len(eps))]
    evaluate_LFKF = [[0 for t in range(times)] for e in range(len(eps))]
    size_GMM = [[0 for t in range(times)] for e in range(len(eps))]
    construction_time_coreset = [[0 for t in range(times)] for e in range(len(eps))]
    opt_time_coreset = [[0 for t in range(times)] for e in range(len(eps))]
    construction_time_uniform = [[0 for t in range(times)] for e in range(len(eps))]
    opt_time_uniform = [[0 for t in range(times)] for e in range(len(eps))]
    construction_time_LFKF = [[0 for t in range(times)] for e in range(len(eps))]
    opt_time_LFKF = [[0 for t in range(times)] for e in range(len(eps))]

    # compute the optimal clustering value on the full dataset
    start = time.time()
    evaluate_data = opt_GMM(time_series, k, lam_l, lam_u)
    print('opt:')
    opt_time = time.time() - start
    print('opt_value:', evaluate_data)
    print('time:', time.time() - start)

    for e in range(len(eps)):
        print("eps:", eps[e])
        n_sample = int(k * (d - 1) / eps[e] )
        t_sample = int(3 * (d - 1) / eps[e] )
        for t in range(times):
            print("times:", t)
            print('opt_value:', evaluate_data)
            # coreset construction
            print("--------------------Coreset-----------------")
            start = time.time()
            c_GMM, size = coreset_GMM(time_series, k, n_sample, t_sample)
            construction_time_coreset[e][t] = time.time() - start
            size_GMM[e][t] = size
    
            # compute the optimal GMM clustering value on the coreset
            start = time.time()
            evaluate_coreset[e][t] = opt_GMM_coreset(time_series, c_GMM, k, lam_l, lam_u)
            opt_time_coreset[e][t] = time.time() - start + construction_time_coreset[e][t]
            print('time:', time.time() - start + construction_time_coreset[e][t])
            print("--------------------Coreset END-----------------")
    
            # uniform construction
            print("--------------------Uniform-----------------")
            start = time.time()
            u_GMM = uniform(time_series, size)
            construction_time_uniform[e][t] = time.time() - start
    
            # compute the optimal GMM clustering value on the uniform sample
            start = time.time()
            evaluate_uniform[e][t] = opt_GMM_coreset(time_series, u_GMM, k, lam_l, lam_u)
            opt_time_uniform[e][t] = time.time() - start + construction_time_uniform[e][t]
            print('time:', time.time() - start + construction_time_uniform[e][t])
            print("--------------------Uniform END-----------------")

            # LFKF construction
            print("--------------------LFKF-----------------")
            start = time.time()
            LFKF_GMM = LFKF(time_series, k, size)
            construction_time_LFKF[e][t] = time.time() - start

            # compute the optimal GMM clustering value on the LFKF sample
            start = time.time()
            evaluate_LFKF[e][t] = opt_GMM_coreset(time_series, LFKF_GMM, k, lam_l, lam_u)
            opt_time_LFKF[e][t] = time.time() - start + construction_time_LFKF[e][t]
            print('time:', time.time() - start + construction_time_LFKF[e][t])
            print("--------------------LFKF END-----------------")

    size_GMM_stat = [[np.mean(size_GMM[e]), np.std(size_GMM[e])] for e in range(len(eps))]
    # coreset
    evaluate_coreset_stat = [[np.mean(evaluate_coreset[e]), np.std(evaluate_coreset[e])] for e in range(len(eps))]
    construction_time_coreset_stat = [[np.mean(construction_time_coreset[e]), np.std(construction_time_coreset[e])] for e in range(len(eps))]
    opt_time_coreset_stat = [[np.mean(opt_time_coreset[e]), np.std(opt_time_coreset[e])] for e in range(len(eps))]
    # uniform
    evaluate_uniform_stat = [[np.mean(evaluate_uniform[e]), np.std(evaluate_uniform[e])] for e in range(len(eps))]
    construction_time_uniform_stat = [[np.mean(construction_time_uniform[e]), np.std(construction_time_uniform[e])] for e in range(len(eps))]
    opt_time_uniform_stat = [[np.mean(opt_time_uniform[e]), np.std(opt_time_uniform[e])] for e in range(len(eps))]
    # LFKF
    evaluate_LFKF_stat = [[np.mean(evaluate_LFKF[e]), np.std(evaluate_LFKF[e])] for e in range(len(eps))]
    construction_time_LFKF_stat = [[np.mean(construction_time_LFKF[e]), np.std(construction_time_LFKF[e])] for e in range(len(eps))]
    opt_time_LFKF_stat = [[np.mean(opt_time_LFKF[e]), np.std(opt_time_LFKF[e])] for e in range(len(eps))]
    
    return eps, evaluate_data, opt_time, evaluate_coreset_stat, evaluate_uniform_stat, evaluate_LFKF_stat, size_GMM_stat, construction_time_coreset_stat, opt_time_coreset_stat, construction_time_uniform_stat, \
           opt_time_uniform_stat, construction_time_LFKF_stat, opt_time_LFKF_stat

#############################################