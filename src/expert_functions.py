import numpy as np
import pandas as pd
from src.sem_functions import sem_fit_no_var


def expert_data(votes, one_hot):
    excl_experts_data = []

    for i in range(votes.shape[1]-1):
        print('df ', i, ' done.')
        e = votes[:,i]+1
        e_values = np.max(e)
        e_one_hot = np.eye(e_values)[e-1]
        excl_e = one_hot - e_one_hot

        excl_experts_data.append(excl_e)

    return excl_experts_data


def expert_sem(data, K, rtol=1e-5, max_iter=20, restarts=20):
    excl_experts_sem = []

    for i in range(len(data)):
        print('exclude expert ',i)
        df = data[i]
        sem = sem_fit_no_var(df, K=K, max_iter=max_iter, rtol=rtol, restarts=restarts)
        #sem = sem_fit_no_var(df, K=17, max_iter=10, rtol=1e-3, restarts=10)
        excl_experts_sem.append(sem)

    return excl_experts_sem

# calculate modified chi_j^2

def chi2(tau, obs, all_cities, classes_num = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])):
    chi2_list = []
    for e in range(len(tau)):

        e_votes = all_cities[obs,e]
        counts = np.array([(e_votes == k).sum() for k in classes_num])
        expected = np.mean(tau[e],axis=0)*len(e_votes)
        chi2_j_k = ((counts - expected)**2)/expected
        chi2_j = np.sum(chi2_j_k)
        chi2_list.append(chi2_j)

    return chi2_list


# calculate expected distribution of chi2
def sim_chi2(tau_experts, tau_all, num_S = 500,
             classes_num = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])):

    sim_chi2_list = []
    for tau in tau_experts:

        sim_chi2_list_j = []

        w = []

        for i in range(len(tau_all)):
            w_i = np.argmax(np.random.multinomial(1, tau_all[i],size=num_S),axis=1)+1
            w.append(w_i)

        sim_votes_all = pd.DataFrame(w)

        for s in range(num_S):
            sim_votes = sim_votes_all[s]
            sim_counts = np.array([(sim_votes == k).sum() for k in classes_num]).T
            expected = np.mean(tau,axis=0)*len(sim_votes) # what to use here?
            # We compare the simulated votes against the posterior?
            sim_chi2_k = ((sim_counts - expected)**2)/expected
            sim_chi2 = np.sum(sim_chi2_k)
            sim_chi2_list_j.append(sim_chi2)

        sim_chi2_list.append(sim_chi2_list_j)

    return sim_chi2_list


# test heterogeneity of the experts

def expert_heterogeneity(all_cities, city_data_full, tau_all, classes, num_S, K):

    excl_experts_data = expert_data(votes=all_cities, one_hot=city_data_full)
    excl_experts_sem = expert_sem(data=excl_experts_data, K=K)
    excl_experts_tau = []
    for exp in excl_experts_sem:
        excl_experts_tau.append(exp[4])

    chi2_list = chi2(tau = excl_experts_tau, classes_num = classes)
    sim_chi2_list = sim_chi2(tau_experts = excl_experts_tau, tau_all = tau_all, num_S = num_S,
                             classes_num = classes)

    # comparison and p-value calculation:
    p_vals = []

    for e in range(len(sim_chi2_list)):
        p = np.sum(sim_chi2_list[e]>chi2_list[e])/len(sim_chi2_list[e])
        p_vals.append(p)

    return excl_experts_tau, chi2_list, sim_chi2_list, p_vals


# different variant of sim_chi2 (all experts compared against one distribution)

def sim_chi2_v2(tau_all, num_S = 500,
                classes_num = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])):

    sim_chi2_list = []

    w = []
    for i in range(len(tau_all)):
        w_i = np.argmax(np.random.multinomial(1, tau_all[i],size=num_S),axis=1)+1
        w.append(w_i)

    sim_votes_all = pd.DataFrame(w)

    for s in range(num_S):
        sim_votes = sim_votes_all[s]
        sim_counts = np.array([(sim_votes == k).sum() for k in classes_num]).T
        expected = np.mean(tau_all,axis=0)*len(sim_votes) # what to use here?
        # We compare the simulated votes against the posterior?
        sim_chi2_k = ((sim_counts - expected)**2)/expected
        sim_chi2 = np.sum(sim_chi2_k)
        sim_chi2_list.append(sim_chi2)


    return sim_chi2_list