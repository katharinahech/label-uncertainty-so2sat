import numpy as np
from scipy.linalg import sqrtm
from statsmodels.stats import weightstats as stests
import pandas as pd

from src.sem_functions import sem_fit_var



def sem_fit_var_per_city(city, data, K, max_iter=50,rtol=1e-3, restarts=3):

    city_data = data[data.City==city]
    city_data = np.array(city_data.drop('City', axis=1))
    best_loss, best_pi, best_theta, best_theta_original, best_tau, theta_var, matching, z_vals = sem_fit_var(city_data, K, max_iter=max_iter,
                                                                                                             rtol=rtol, restarts=restarts)

    return city, best_loss, best_pi, best_theta, best_theta_original, best_tau, theta_var, matching, z_vals


def theta_comparison_city(city_sem, city_list, K):
    city_test_results = []

    for i in range(len(city_list)):
        for j in range(len(city_list)):
            if j>i:
                print(city_sem[i][0], ' vs. ', city_sem[j][0])

                c1 = city_sem[i]
                c2 = city_sem[j]

                # get vectorization of city thetas
                c1_vec = c1[3][:,0:(K-1)].flatten()
                c2_vec = c2[3][:,0:(K-1)].flatten()
                diff_vec = c1_vec - c2_vec

                # get variance of vectorization
                c1_vec_var = c1[6]
                c2_vec_var = c2[6]

                # Under H0: E(diff_vec) = 0 and Var(diff_vec) = Var(b_vec) + Var(m_vec)
                diff_vec_var = c1_vec_var + c2_vec_var
                sqrt_diff_var = sqrtm(diff_vec_var)
                sqrt_inf_diff_var = np.linalg.pinv(sqrt_diff_var)
                T = np.matmul(sqrt_inf_diff_var, diff_vec)
                test = stests.ztest(T, x2=None, value=0)

                # test results
                result = (city_sem[i][0], city_sem[j][0], test[1])
                city_test_results.append(result)

        # p values
        city_p = pd.DataFrame(city_test_results, columns =['C1', 'C2', 'p'])

    return city_test_results, city_p