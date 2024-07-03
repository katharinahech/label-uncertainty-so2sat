import numpy as np
import pandas as pd
from scipy.stats import dirichlet, multinomial
from scipy.linalg import block_diag, sqrtm
from sklearn.cluster import KMeans


def compute_log_likelihood(Y_test, pi, theta):
    mn_probs = np.zeros(Y_test.shape[0])
    for k in range(theta.shape[0]):
        mn_probs_k = get_mixture_weight(pi, k) * multinomial_prob(Y_test, theta[k])
        mn_probs += mn_probs_k
    mn_probs[mn_probs == 0] = np.finfo(float).eps
    return np.log(mn_probs).sum()

def multinomial_prob(counts, theta_k, log=False):
    """
    Evaluates the multinomial probability for a given vector of counts
    counts: (N x C), matrix of counts
    beta: (C), vector of multinomial parameters for a specific cluster k
    Returns:
    p: (N), scalar values for the probabilities of observing each count vector given the beta parameters
    """
    if np.sum(theta_k)>1:
        theta_k = theta_k/np.sum(theta_k)

    #n = counts[0].sum(axis=-1)
    n = counts.sum(axis=-1)
    m = multinomial(n, theta_k)
    if log:
        return m.logpmf(counts)
    return m.pmf(counts)

def e_step(Y, pi, theta):
    """
    Performs E-step on MNMM model
    Each input is numpy array:
    Y: (N x C), matrix of counts
    pi: (K) or (NxK) in the case of individual weights, mixture component weights
    theta: (K x C), multinomial categories weights
    Returns:
    tau: (N x K), posterior probabilities for objects clusters assignments
    """
    N = Y.shape[0]
    K = theta.shape[0]
    posterior_prob = np.zeros((N, K))

    for k in range(K):
        p_k = [np.log(pi[k]) + np.sum(np.log(theta[k]**v)) for v in Y]
        posterior_prob[:,k] = np.exp(p_k).reshape(-1,)

    # To avoid division by 0
    posterior_prob[posterior_prob == 0] = np.finfo(float).eps

    denum = posterior_prob.sum(axis=1)
    tau = posterior_prob / denum.reshape(-1, 1)

    return tau

def e_step_nozerocorrection(Y, pi, theta):
    """
    Performs E-step on MNMM model
    Each input is numpy array:
    Y: (N x C), matrix of counts
    pi: (K) or (NxK) in the case of individual weights, mixture component weights
    theta: (K x C), multinomial categories weights
    Returns:
    tau: (N x K), posterior probabilities for objects clusters assignments
    """
    N = Y.shape[0]
    K = theta.shape[0]
    posterior_prob = np.zeros((N, K))
    theta[theta == 0] = np.finfo(float).eps

    for k in range(K):
        p_k = [np.log(pi[k]) + np.sum(np.log(theta[k]**v)) for v in Y]
        posterior_prob[:,k] = np.exp(p_k).reshape(-1,)

    # To avoid division by 0
    # posterior_prob[posterior_prob == 0] = np.finfo(float).eps

    denum = posterior_prob.sum(axis=1)
    tau = posterior_prob / denum.reshape(-1, 1)
    #tau[tau == 0] = np.finfo(float).eps

    return tau


def get_mixture_weight(pi, k):
    return pi[k]

def m_step(Y, tau):
    """
    Performs M-step on MNMM model
    Each input is numpy array:
    Y: (N x C), matrix of counts
    tau: (N x K), posterior probabilities for objects clusters assignments
    Returns:
    pi: (K), mixture component weights
    theta: (K x C), mixture categories weights
    """
    # Compute alpha
    pi = m_step_pi(tau)

    # Compute beta
    theta = m_step_theta(Y, tau)

    return pi, theta

def m_step_pi(tau):
    pi = tau.sum(axis=0) / tau.sum()
    return pi

def m_step_theta(Y, tau):
    weighted_counts = tau.T.dot(Y)
    theta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)
    return theta

def compute_vlb(Y, pi, theta, tau):
    """
    Computes the variational lower bound
    Y: (N x C), data points
    pi: (K) or (NxK) with individual weights, mixture component weights
    theta: (K x C), multinomial categories weights
    tau: (N x K), posterior probabilities for objects clusters assignments
    Returns value of variational lower bound
    """
    loss = 0
    for k in range(theta.shape[0]):
        weights = tau[:, k]
        mnp = multinomial_prob(Y, theta[k])
        mnp_ex = np.delete(mnp, np.where(mnp==0))
        weights_ex = np.delete(weights, np.where(mnp==0))
        loss += np.sum(weights_ex * (np.log(pi[k]) + np.log(mnp_ex)))
        loss -= np.sum(weights_ex * np.log(weights_ex))
        #loss += np.sum(weights * (np.log(pi[k]) + multinomial_prob(Y, theta[k], log=True)))
        #loss -= np.sum(weights * np.log(weights))
    return loss

def init_params(Y,K):
    kmeans = KMeans(init="random",n_clusters=K,n_init=10,max_iter=300,random_state=42)
    kmeans.fit(Y)

    # extract params
    z_hat_init = kmeans.labels_
    z_sums = np.unique(z_hat_init,return_counts=True)[1]
    pi_new = z_sums/len(z_hat_init)

    # apply matching
    votes_prop = np.sum(Y,axis=0)/(12*len(Y))
    matches = []
    matches_init = {}
    keys = np.argsort(pi_new)[::-1]

    C=K+1
    for p in keys:
        k=1
        while k in range(1,C+1):
            m = np.argsort(votes_prop)[-k]

            if m not in matches:
                matches.append(m)
                break
            elif k==17:
                break
            else:
                k=k+1

    for i in range(K):
        matches_init[keys[i]] = matches[i]

    # match pi
    df_pi = pd.DataFrame(pi_new).T
    df_pi.rename(columns=matches_init, inplace=True)
    df_pi_matched = df_pi.T.sort_index()
    pi_matched_init = np.concatenate(np.array(df_pi_matched))

    df_z = pd.DataFrame(z_hat_init, columns=['z'])
    df_z = df_z.replace({'z': matches_init})
    z_matched_init = np.concatenate(np.array(df_z))

    # calculate theta
    theta_init = np.zeros((K,K))
    for l in range(K):
        theta_l = []
        for k in range(K):
            ind_z = [1 if z==l else 0 for z in z_matched_init]
            ind_v = [v/12 if v!=0 else 0 for v in Y[:,k]]
            theta_l.append(np.sum(np.array(ind_z)*np.array(ind_v))/np.sum(ind_z))
        theta_init[l] = theta_l

    return pi_matched_init, theta_init

def sem_step(tau):
    prob_hat_list = [np.random.multinomial(1,x,size=1) for x in tau]
    prob_hat = np.concatenate(prob_hat_list, axis=0)
    lcz_hat = np.sum(prob_hat,axis=0)

    return prob_hat, lcz_hat

def sem_m_step(Y, prob_hat, lcz_hat,K):
    """
    Performs M-step on MNMM model
    Each input is numpy array:
    Y: (N x C), matrix of counts
    tau: (N x K), posterior probabilities for objects clusters assignments
    Returns:
    pi: (K), mixture component weights
    theta: (K x C), mixture categories weights
    """
    # Compute pi
    pi = sem_step_pi(Y, lcz_hat)

    # Compute theta
    theta = sem_step_theta(Y, prob_hat)

    return pi, theta

def sem_step_pi(Y, lcz_hat):
    pi = lcz_hat/Y.shape[0]
    return pi

def sem_step_theta(Y, prob_hat):
    z = prob_hat.T.dot(Y/np.max(Y))
    theta=z/z.sum(axis=-1).reshape(-1,1)
    return theta

# functions for including label switching in SEM

def get_permutations_bayesian_sem(theta, pi, data):
    theta = theta
    K = len(theta)
    diag_pi = np.diag(pi)
    z_y = pd.DataFrame(np.matmul(theta, diag_pi))
    n_experts = np.sum(data,axis=1)[0]
    votes = pd.DataFrame(np.sum(data/n_experts, axis=0),columns=['votes'])
    df_votes = pd.concat([z_y,votes.T],axis=0)
    df_votes_sorted = df_votes.T.sort_values('votes', ascending=False).T

    theta = pd.DataFrame(theta)
    theta_votes = pd.concat([theta,votes.T],axis=0)
    theta_votes_sorted = df_votes.T.sort_values('votes', ascending=False).T

    # get permutation
    voted_class = []
    modes = []
    for i in np.array(df_votes_sorted.columns):
        k = 1
        while k in range(1,K+1):
            b = np.array(df_votes.T)[:,0:K][i]
            m = np.argsort(b)[-k]
            if m not in modes:
                modes.append(m)
                voted_class.append(i)
                break
            else:
                k=k+1

    # combine with "mode"
    modes = pd.DataFrame(modes, columns=['modes'], index=voted_class)
    city_permuted = pd.concat([theta_votes,modes.T],axis=0)

    # combine with marginal probabilities "pi"
    pi = pd.DataFrame(pi, columns=['pi'])
    city_permuted_pi = pd.concat([city_permuted, pi],axis=1)

    # second matching round by number of votes

    df_sorted = city_permuted.T.sort_values('votes', ascending=False).T
    df_sorted = df_sorted.drop(['votes', 'modes'],axis=0)

    df = np.array(df_sorted)

    c = pd.DataFrame(np.around(df,3))

    return c

def get_final_thetas_sem(permuted_theta):
    theta = permuted_theta
    K = len(theta)
    correct_index = pd.DataFrame(theta.columns[0:K], columns = ['lcz_label'])
    theta = pd.concat([correct_index, theta], axis=1)
    matrix_renamed = theta.set_index(['lcz_label'])

    matrix_renamed = matrix_renamed.drop(['pi'],axis=1)
    final_theta = matrix_renamed[matrix_renamed.index.notnull()]

    return final_theta


def get_matching_bayesian(theta, pi, data):
    theta = theta
    K = len(theta)
    diag_pi = np.diag(pi)
    z_y = pd.DataFrame(np.matmul(theta, diag_pi))
    n_experts = np.sum(data,axis=1)[0]
    votes = pd.DataFrame(np.sum(data/n_experts, axis=0),columns=['votes'])
    df_votes = pd.concat([z_y,votes.T],axis=0)
    df_votes_sorted = df_votes.T.sort_values('votes', ascending=False).T

    theta = pd.DataFrame(theta)
    theta_votes = pd.concat([theta,votes.T],axis=0)
    theta_votes_sorted = df_votes.T.sort_values('votes', ascending=False).T


    # get permutation
    voted_class = []
    modes = []
    for i in np.array(df_votes_sorted.columns):
        k = 1
        while k in range(1,K+1):
            b = np.array(df_votes.T)[:,0:K][i]
            m = np.argsort(b)[-k]
            if m not in modes:
                modes.append(m)
                voted_class.append(i)
                break
            else:
                k=k+1

    # get matching between cluster label and voted class
    matching = pd.DataFrame(modes, columns=['clusters'], index=voted_class)

    return matching


def get_permuted_params(matching, theta, pi, tau):
    matching = matching
    renamed_index = matching.sort_values(by='clusters').index

    # theta
    theta_full_df = pd.DataFrame(theta)
    theta_full_df.index = renamed_index
    theta_full_df = theta_full_df.sort_index(axis=0)
    theta_permuted = np.array(theta_full_df)

    # pi

    pi_full_df = pd.DataFrame(pi)
    pi_full_df.index = renamed_index
    pi_full_df = pi_full_df.sort_index(axis=0)
    pi_permuted = np.array(pi_full_df)

    # tau

    tau_full_df = pd.DataFrame(tau)
    tau_full_df.columns = renamed_index
    tau_full_df = tau_full_df.sort_index(axis=1)
    tau_permuted = np.array(tau_full_df)

    return theta_permuted, pi_permuted, tau_permuted, matching

# rewrite function of sem to consider z-values and calculate variances

def sem_train_once_var(Y, K, max_iter=100, rtol=1e-1):
    '''
    Runs one full cycle of the EM algorithm
    :param Y: (N, C), matrix of counts
    :return: The best parameters found along with the associated loss
    '''
    loss = float('inf')
    pi, theta = init_params(Y, K)
    theta_original = theta
    tau = None

    block_iter = []
    theta_iter = []

    z = []

    for it in range(max_iter):
        prev_loss = loss
        prev_pi = pi
        prev_theta = theta
        prev_theta_original = theta_original
        prev_tau = tau
        tau = e_step(Y, pi, theta)
        prob_hat, lcz_hat = sem_step(tau)
        pi, theta = sem_m_step(Y, prob_hat, lcz_hat, K)
        loss = compute_vlb(Y, pi, theta, tau)

        # add output of estimated z
        z_iter = np.argmax(prob_hat,axis=1)
        z.append(z_iter)

        # shuffle cluster labels to solve label switching:
        matching = get_matching_bayesian(theta, pi, Y)
        theta_permuted, pi_permuted, tau_permuted, matching = get_permuted_params(matching, theta, pi, tau)
        theta, pi, tau = theta_permuted, pi_permuted, tau_permuted

        print('Loss: %f' % loss)
        if it > 1 and np.abs((prev_loss - loss) / prev_loss) < rtol:
            break
        if it > 1 and np.isnan(loss):
            loss = prev_loss
            theta = prev_theta
            theta_original = prev_theta_original
            pi = prev_pi
            tau = prev_tau
            break
        if it <= 1 and np.isnan(loss):
            break


        # compute variance components:

        dim = theta.shape[0]*(theta.shape[1]-1)

        var_theta_ls = []
        for i in range(len(theta)):
            theta_l = np.array([theta[i][:-1]])
            theta_l_T = np.transpose(theta_l)
            prod_l = np.diag(theta[i][:-1]) - np.matmul(theta_l_T, theta_l) # zähler
            var_theta_l = prod_l/lcz_hat[i]
            var_theta_ls.append(var_theta_l)

        block = block_diag()
        for x in var_theta_ls:
            block = block_diag(block, x)

        block = np.delete(block, 0, axis=0)
        block_iter.append(block)
        theta_iter.append(np.delete(theta, -1, axis=1))


    if np.isnan(loss):
        var_theta = float("nan")
        return loss, pi, theta, theta_original, tau, var_theta, matching, z

    else:
        # compute overall variance:

        # p1
        block_sum = sum(block_iter)
        p1 = block_sum/len(block_iter)

        # p2
        theta_mean = np.nanmean(theta_iter,axis=0).flatten()
        theta_mean = theta_mean.reshape((dim,1))
        prod = []
        for b in theta_iter:
            b_arr = np.array(b)
            b_vec = b_arr.flatten()
            b_vec = b_vec.reshape((dim, 1))
            x = b_vec - theta_mean
            x_t = np.transpose(x)
            prod.append(np.dot(x,x_t))

        p2 = np.sum(prod,axis=0)/(len(theta_iter)-1)

        # variance
        var_theta = p1 + p2


    return loss, pi, theta, theta_original, tau, var_theta, matching, z
    #return loss, pi, theta, tau, var_theta, z

def sem_fit_var(Y, K, restarts=10, max_iter=100, rtol=1e-1):
    '''
    Starts with random initialization *restarts* times
    Runs optimization until saturation with *rtol* reached
    or *max_iter* iterations were made.
    :param Y: (N, C), matrix of counts
    :return: The best parameters found along with the associated loss
    '''
    best_loss = -float('inf')
    best_pi = None
    best_theta = None
    best_theta_original = None
    best_tau = None
    theta_var = None

    z_vals = []

    for it in range(restarts):
        print('iteration %i' % it)
        loss, pi, theta, theta_original, tau, var, matching, z = sem_train_once_var(Y, K, max_iter, rtol)

        #loss, pi, theta, tau, var, z = sem_train_once_var(Y, K, max_iter, rtol)
        z_vals.append(z)
        if loss > best_loss:
            print('better loss on iteration %i: %.10f' % (it, loss))
            best_loss = loss
            best_pi = pi
            best_theta = theta
            best_theta_original = theta_original
            best_tau = tau
            theta_var = var

    return best_loss, best_pi, best_theta, best_theta_original, best_tau, theta_var, matching, z_vals
    #return best_loss, best_pi, best_theta, best_tau, theta_var, z_vals

def run_sem_city_var(city, data, K):
    city_data = data[data.City==city]
    city_data = np.array(city_data.drop('City', axis=1))
    best_loss, best_pi, best_theta, best_theta_original, best_tau, theta_var, matching, z_vals = sem_fit_var(city_data, K, max_iter=50, rtol=1e-3, restarts=10)
    return city, best_loss, best_pi, best_theta, best_theta_original, best_tau, theta_var, matching, z_vals

def cm_to_inch(value):
    return value/2.54

# versions without variance for faster execution

def sem_train_once_no_var(Y, K, max_iter=100, rtol=1e-1):
    '''
    Runs one full cycle of the EM algorithm
    :param Y: (N, C), matrix of counts
    :return: The best parameters found along with the associated loss
    '''
    loss = float('inf')
    pi, theta = init_params(Y, K)
    theta_original = theta
    tau = None


    for it in range(max_iter):
        prev_loss = loss
        prev_pi = pi
        prev_theta = theta
        prev_theta_original = theta_original
        prev_tau = tau
        tau = e_step(Y, pi, theta)
        prob_hat, lcz_hat = sem_step(tau)
        pi, theta = sem_m_step(Y, prob_hat, lcz_hat, K)
        loss = compute_vlb(Y, pi, theta, tau)

        # shuffle cluster labels to solve label switching:
        matching = get_matching_bayesian(theta, pi, Y)
        theta_permuted, pi_permuted, tau_permuted, matching = get_permuted_params(matching, theta, pi, tau)
        theta, pi, tau = theta_permuted, pi_permuted, tau_permuted

        print('Loss: %f' % loss)
        if it > 0 and np.abs((prev_loss - loss) / prev_loss) < rtol:
            break
        if it > 0 and np.isnan(loss):
            loss = prev_loss
            theta = prev_theta
            theta_original = prev_theta_original
            pi = prev_pi
            tau = prev_tau
            break


    return loss, pi, theta, theta_original, tau, matching
    # return loss, pi, theta, tau

def sem_fit_no_var(Y, K, restarts=10, max_iter=20, rtol=1e-5):
    '''
    Starts with random initialization *restarts* times
    Runs optimization until saturation with *rtol* reached
    or *max_iter* iterations were made.
    :param Y: (N, C), matrix of counts
    :return: The best parameters found along with the associated loss
    '''
    best_loss = -float('inf')
    best_pi = None
    best_theta = None
    best_theta_original = None
    best_tau = None

    for it in range(restarts):
        print('iteration %i' % it)
        loss, pi, theta, theta_original, tau, matching = sem_train_once_no_var(Y, K, max_iter, rtol)
        #loss, pi, theta, tau = sem_train_once_no_var(Y, K, max_iter, rtol)
        if loss > best_loss:
            print('better loss on iteration %i: %.10f' % (it, loss))
            best_loss = loss
            best_pi = pi
            best_theta = theta
            best_theta_original = theta_original
            best_tau = tau

    return best_loss, best_pi, best_theta, best_theta_original, best_tau, matching
    #return best_loss, best_pi, best_theta, best_tau
