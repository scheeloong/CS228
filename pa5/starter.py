# Learning: GMM and EM
# Authors: Gunaa A V, Isaac C, Volodymyr K, Haque I, Aditya G
# Last updated: March 3, 2017

import numpy as np
from pprint import pprint
import copy
import math
import time
import matplotlib.pyplot as plt
import pdb
from scipy.stats import multivariate_normal
import itertools

LABELED_FILE = "surveylabeled.dat"
UNLABELED_FILE = "surveyunlabeled.dat"


#===============================================================================
# General helper functions

def colorprint(message, color="rand"):
    """Prints your message in pretty colors! 

    So far, only the colors below are available.
    """
    if color == 'none': print message; return
    if color == 'demo':
        for i in range(99):
            print '%i-'%i + '\033[%sm'%i + message + '\033[0m\t',
    print '\033[%sm'%{
        'neutral' : 99,
        'flashing' : 5,
        'underline' : 4,
        'magenta_highlight' : 45,
        'red_highlight' : 41,
        'pink' : 35,
        'yellow' : 93,   
        'teal' : 96,     
        'rand' : np.random.randint(1,99),
        'green?' : 92,
        'red' : 91,
        'bold' : 1
    }.get(color, 1)  + message + '\033[0m'

def read_labeled_matrix(filename):
    """Read and parse the labeled dataset.

    Output:
        Xij: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        Zij: dictionary of party choices.
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a float.
        N, M: Counts of precincts and voters.
    """
    Zij = {} 
    Xij = {}
    M = 0.0
    N = 0.0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            i, j, Z, X1, X2 = line.split()
            i, j = int(i), int(j)
            if i>N: N = i
            if j>M: M = j

            Zij[i-1, j-1] = float(Z)
            Xij[i-1, j-1] = np.matrix([float(X1), float(X2)])
    return Xij, Zij, N, M

def read_unlabeled_matrix(filename):
    """Read and parse the unlabeled dataset.
    
    Output:
        Xij: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters.
    """
    Xij = {}
    M = 0.0
    N = 0.0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            i, j, X1, X2 = line.split()
            i, j = int(i), int(j)
            if i>N: N = i
            if j>M: M = j

            Xij[i-1, j-1] = np.matrix([float(X1), float(X2)])
    return Xij, N, M


#===============================================================================
# Functions that define the probability distribution
#
# There are helper functions that you may find useful for solving the problem.
# You do not neet to use them, but we have found them to be helpful.
# Not all of them are implemented. We give a few examples, and you should 
# implement the rest. We suggest you first start by implementing the rest of the
# homework and fill-in the missing functions here when you realize that you need
# them.

def p_yi(y_i, phi):
    """Probability of y_i.

    Bernouilli distribution with parameter phi.
    """
    return (phi**y_i) * ((1-phi)**(1-y_i))

def p_zij(z_ij, pi):
    """Probability of z_ij.

    Bernouilli distribution with parameter pi.
    """
    return (pi**z_ij) * ((1-pi)**(1-z_ij))

def p_zij_given_yi(z_ij, y_i, lambd):
    """Probability of z_ij given yi.

    Bernouilli distribution with parameter lambd that targets
    the variable (z_ij == y_i).
    """
    if z_ij == y_i:
        return lambd
    return 1-lambd

def z_marginal(z_ij, lambd, phi):
    """Marginal probability of z_ij with yi marginalized out."""
    return p_zij_given_yi(z_ij, 1, lambd) * p_yi(1, phi) \
         + p_zij_given_yi(z_ij, 0, lambd) * p_yi(0, phi)

def p_xij_given_zij(x_ij, mu_zij, sigma_zij):
    """Probability of x_ij.

    Given by multivariate normal distribution with params mu_zij and sigma_zij.
    
    Input:
        x_ij: (1,2) array of continuous variables
        mu_zij: (1,2) array representing the mean of class z_ij
        sigma_zij: (2,2) array representing the covariance matrix of class z_ij

    All arrays must be instances of numpy.matrix class.
    """
    assert isinstance(x_ij, np.matrix)
    k = x_ij.shape[1]; assert(k==2)

    det_sigma_zij = sigma_zij[0, 0]*sigma_zij[1, 1] - sigma_zij[1, 0]*sigma_zij[0, 1]
    assert det_sigma_zij > 0

    sigma_zij_inv = -copy.copy(sigma_zij); sigma_zij_inv[0, 0] = sigma_zij[1, 1]; sigma_zij_inv[1, 1] = sigma_zij[0, 0]
    sigma_zij_inv /= det_sigma_zij

    # print "better be identity matrix:\n", sigma_zij.dot(sigma_zij_inv)

    multiplicand =  (((2*math.pi)**k)*det_sigma_zij)**(-0.5)
    exponentiand = -.5 * (x_ij-mu_zij).dot(sigma_zij_inv).dot((x_ij-mu_zij).T)
    exponentiand = exponentiand[0,0]
    return multiplicand * np.exp(exponentiand)

def p_zij_given_xij_unnorm(z_ij, x_ij, lambd, phi, mu_zij, sigma_zij):
    """Unnormalized posterior probability of z_ij given x_ij."""

    # -------------------------------------------------------------------------
    # TODO (Optional): Put your code here
    
    pass

    # END_YOUR_CODE
    

def p_xij_given_yi(x_ij, y_i, mu_0, sigma_0, mu_1, sigma_1, lambd):
    """Probability of x_ij given y_i.
    
    To compute this, marginalize (i.e. sum out) over z_ij.
    """

    # -------------------------------------------------------------------------
    # TODO (Optional): Put your code here
    
    pass

    # END_YOUR_CODE
    

def p_yi_given_xij_unnorm(x_ij, y_i, mu_0, sigma_0, mu_1, sigma_1, phi, lambd):
    """Unnormalized posterior probability of y_ij given x_ij.
    
    Hint: use Bayes' rule!
    """

    # -------------------------------------------------------------------------
    # TODO (Optional): Put your code here
    
    pass

    # END_YOUR_CODE


def estimate(Xij, Zij):
    pi = 0.0
    # -------------------------------------------------------------------------
    # TODO: Code to compute pi

    cnt = 0.0
    num = 0.0
    for key in Zij:
        num+=Zij[key]
        cnt+=1.0

    pi = num/cnt

    # END_YOUR_CODE

    mean = [np.matrix([0.0, 0.0]), np.matrix([0.0, 0.0])]
    cnt = [0.0,0.0]
    # -------------------------------------------------------------------------
    # TODO: Code to compute mean0, mean1 

    for key in Zij:
        z = int(Zij[key])
        mean[z] = np.add(mean[z], Xij[key])
        cnt[z] += 1.0

    mean[0]/=cnt[0]
    mean[1]/=cnt[1]



    sigma = [np.matrix([[0.0,0.0],[0.0,0.0]]), np.matrix([[0.0,0.0],[0.0,0.0]])]
    # -------------------------------------------------------------------------
    # TODO: Code to compute sigma0, sigma1 

    for key in Zij:

        z = int(Zij[key])
        inside = np.subtract(Xij[key], mean[z])
        sigma[z] = np.add(sigma[z], np.outer(inside,inside))

    sigma[0]/=cnt[0]
    sigma[1]/=cnt[1]
    # END_YOUR_CODE

    return pi, mean[0], mean[1], sigma[0], sigma[1]


def MLE_Estimation():
    """Perform MLE estimation of Model A parameters.

    Output:
        pi: (float), estimate of party proportions
        mean0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
        mean1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
        sigma0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
        sigma1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1
    """
    
    Xij, Zij, N, M = read_labeled_matrix(LABELED_FILE)
    return estimate(Xij, Zij)
    
def perform_em_modelA(X, N, M, init_params, max_iters=50, eps=1e-2):
    """Estimate Model A paramters using EM

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        init_params: parameters of the model given as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 

    Output:
        params: parameters of the trained model encoded as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']
        log_likelihood: array of log-likelihood values across iterations
    """

    def compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, pi):
        """Compute the log-likelihood of the data given our parameters

        Input:
            pi: (float), estimate of party proportions
            mu_0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
            mu_1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
            sigma_0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
            sigma_1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1

        Output:
            ll: (float), value of the log-likelihood
        """
        ll = 0.0
        # print "S"
        # print mu_0
        # print mu_1
        # print sigma_0
        # print sigma_1
        # print "D"

        #G = [multivariate_normal(mean=mu_0.A1, cov=sigma_0.A), multivariate_normal(mean=mu_1.A1, cov=sigma_1.A)]

        for key in X:
            ll += np.log((1.-pi)*p_xij_given_zij(X[key], mu_0, sigma_0) + pi*p_xij_given_zij(X[key], mu_1, sigma_1))
        # -------------------------------------------------------------------------
        # TODO: Code to compute ll

        # END_YOUR_CODE
        return ll
    
    # unpack the parameters

    mu_0 = init_params['mu_0']
    mu_1 = init_params['mu_1']
    sigma_0 = init_params['sigma_0']
    sigma_1 = init_params['sigma_1']
    pi = init_params['pi']

    # set up list that will hold log-likelihood over time
    log_likelihood = [compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, pi)]

    for iter in xrange(max_iters):
        # -------------------------------------------------------------------------
        # TODO: Code for the E step

        Z = {}
        for key in X:
            prob_1 = p_xij_given_zij(X[key], mu_1, sigma_1) * pi
            prob_0 = p_xij_given_zij(X[key], mu_0, sigma_0) * (1.-pi)
            Z[key] = prob_1 / (0.0 + prob_0 + prob_1)

        # END_YOUR_CODE

        pi = 0.0
        mu_0 = np.matrix([0.0, 0.0])
        mu_1 = np.matrix([0.0, 0.0])
        sigma_0 = np.matrix([[0.0,0.0],[0.0,0.0]])
        sigma_1 = np.matrix([[0.0,0.0],[0.0,0.0]])

        total = 0.0
        total_0 = 0.0
        total_1 = 0.0

        for key in X:
            pi+=Z[key]
            total+=1.0

            mu_0 += (1.-Z[key]) * X[key]
            mu_1 += Z[key] * X[key]
            
            total_0 += 1.-Z[key]
            total_1 += Z[key]

        pi/=total
        mu_0/=total_0
        mu_1/=total_1

        for key in X:
            inside_0 = np.subtract(X[key], mu_0)
            inside_1 = np.subtract(X[key], mu_1)
            sigma_0 += (1.-Z[key]) * np.outer(inside_0, inside_0)
            sigma_1 += Z[key] * np.outer(inside_1, inside_1)

        
        sigma_0/=total_0
        sigma_1/=total_1

        # -------------------------------------------------------------------------
        # TODO: Code for the M step
        # You should fill the values of pi, mu_0, mu_1, sigma_0, sigma_1

        # END_YOUR_CODE
        
        # Check for convergence
        this_ll = compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, pi)
        log_likelihood.append(this_ll)
        if np.abs((this_ll - log_likelihood[-2]) / log_likelihood[-2]) < eps:
            print "DONE WITH LOG LIKELIHOOD"
            break

    # pack the parameters and return
    params = {}
    params['mu_0'] = mu_0
    params['mu_1'] = mu_1
    params['sigma_0'] = sigma_0
    params['sigma_1'] = sigma_1
    params['pi'] = pi

    return params, log_likelihood

def MLE_of_phi_and_lamdba():
    """Perform MLE estimation of Model B parameters.

    Assumes that Y variables have been estimated using heuristic proposed in
    the question.

    Output:
        MLE_phi: estimate of phi
        MLE_lambda: estimate of lambda
    """
    X, Z, N, M = read_labeled_matrix(LABELED_FILE)
    assert(len(Z.items()) == M*N)

    MLE_phi, MLE_lambda = 0.0, 0.0

    Y = np.zeros(5)

    for i in range(N):
        sig = 0.0
        for j in range(M):
            sig += Z[i,j]

        if sig >= M/2.0:
            Y[i] = 1.0
        else:
            Y[i] = 0.0

    # -------------------------------------------------------------------------
    # TODO: Code to compute MLE_phi, MLE_lambda 
    
    MLE_phi = np.sum(Y) / (0.0 + N)
    
    for i in range(N):
        for j in range(M):
            if Y[i] == Z[i,j]:
                MLE_lambda += 1.0

    MLE_lambda /= (0.0 + N*M)

    # END_YOUR_CODE

    return MLE_phi, MLE_lambda

# one Y
def Y_given_X(X, N, M, params, Yi, YVAL):
    if params == None:
        pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
        MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    else:
        pi = params['pi']
        mu_0 = params['mu_0']
        mu_1 = params['mu_1']
        sigma_0 = params['sigma_0']
        sigma_1 = params['sigma_1']
        MLE_phi = params['phi']
        MLE_lambda = params['lambda']

    pii = [1.-pi, pi]
    mu = [mu_0, mu_1]
    sigma = [sigma_0, sigma_1]
    posterior_y = [None for i in range(N)] 
    # -------------------------------------------------------------------------
    # TODO: Code to compute posterior_y

    top = np.log(MLE_phi)
    bot = np.log(1.0 - MLE_phi)

    for j in range(M):

        pxgy = 0.0
        pxgy_other = 0.0

        for z in range(2):
            pzgy = MLE_lambda if z == YVAL else (1.0-MLE_lambda)
            pxgz = p_xij_given_zij(X[Yi,j], mu[z], sigma[z])

            pxgy += pzgy * pxgz #marginalize the z out
            pxgy_other += pxgz * (1. - pzgy)

        top+=np.log(pxgy)
        bot+=np.log(pxgy_other)

    return math.exp(top) / (math.exp(top) + math.exp(bot))

def YZ_given_X(X, N, M, params, Zi, Zj, ZVAL, YVAL):
    if params == None:
        pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
        MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    else:
        pi = params['pi']
        mu_0 = params['mu_0']
        mu_1 = params['mu_1']
        sigma_0 = params['sigma_0']
        sigma_1 = params['sigma_1']
        MLE_phi = params['phi']
        MLE_lambda = params['lambda']

    pii = [1-pi, pi]
    mu = [mu_0, mu_1]
    sigma = [sigma_0, sigma_1]
    posterior_y = [None for i in range(N)] 
    # -------------------------------------------------------------------------
    # TODO: Code to compute posterior_y

    anst = 0.0
    ansb = 0.0

    for y in range(2):

        top = 1.0
        bot = 1.0
        for j in range(M):

            pxgy = 0.0
            pxgy_other = 0.0

            for z in range(2):
                pzgy = MLE_lambda if z == y else (1.0-MLE_lambda)
                pxgz = p_xij_given_zij(X[Zi,j], mu[z], sigma[z])

                pxgy += pzgy * pxgz #marginalize the z out

            if j == Zj:
                pzgy = MLE_lambda if ZVAL == y else (1.0-MLE_lambda)
                top*=p_xij_given_zij(X[Zi,j], mu[ZVAL], sigma[ZVAL])*pzgy
            else:
                top*=pxgy
            
            bot*=pxgy

        if y == YVAL:
            anst += top
        ansb += bot

    return anst / ansb

def Z_given_X(X, N, M, params, Zi, Zj, ZVAL):
    if params == None:
        pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
        MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    else:
        pi = params['pi']
        mu_0 = params['mu_0']
        mu_1 = params['mu_1']
        sigma_0 = params['sigma_0']
        sigma_1 = params['sigma_1']
        MLE_phi = params['phi']
        MLE_lambda = params['lambda']

    pii = [1-pi, pi]
    mu = [mu_0, mu_1]
    sigma = [sigma_0, sigma_1]
    posterior_y = [None for i in range(N)] 
    # -------------------------------------------------------------------------
    # TODO: Code to compute posterior_y

    anst = 0.0
    ansb = 0.0

    for y in range(2):

        top = 1.0
        bot = 1.0
        for j in range(M):

            pxgy = 0.0
            pxgy_other = 0.0

            for z in range(2):
                pzgy = MLE_lambda if z == y else (1.0-MLE_lambda)
                pxgz = p_xij_given_zij(X[Zi,j], mu[z], sigma[z])

                pxgy += pzgy * pxgz #marginalize the z out

            if j == Zj:
                pzgy = MLE_lambda if ZVAL == y else (1.0-MLE_lambda)
                top*=p_xij_given_zij(X[Zi,j], mu[ZVAL], sigma[ZVAL])*pzgy
            else:
                top*=pxgy
            
            bot*=pxgy

        anst += top
        ansb += bot

    return anst / ansb

    # END_YOUR_CODE

def estimate_leanings_of_precincts(X, N, M, params=None):
    """Estimate the leanings y_i given data X.

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        params: parameters of the model given as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']

    Output:
        Summary: length-N list summarizing the leanings
            Format is: [(i, prob_i, y_i) for i in range(N)]
    """
    if params == None:
        pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
        MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    else:
        pi = params['pi']
        mu_0 = params['mu_0']
        mu_1 = params['mu_1']
        sigma_0 = params['sigma_0']
        sigma_1 = params['sigma_1']
        MLE_phi = params['phi']
        MLE_lambda = params['lambda']

    pii = [1-pi, pi]
    mu = [mu_0, mu_1]
    sigma = [sigma_0, sigma_1]
    posterior_y = [None for i in range(N)] 
    # -------------------------------------------------------------------------
    # TODO: Code to compute posterior_y

    #print N
    #print M

    for i in range(N):
        posterior_y[i] = Y_given_X(X, N, M, params, i, 1)


    # END_YOUR_CODE

    #print posterior_y

    summary = [(i, p, 1 if p>=.5 else 0) for i, p in enumerate(posterior_y)]
    
    print "Y ST"
    for i, num in enumerate(posterior_y):
        print "%s & %s & %s \\\\" % (i, num, "\checkmark" if num > .5 else "")

    print "DONE"

    return summary

# one Z and all X


def plot_individual_inclinations(X, N, M, params=None):
    """Generate 2d plot of inidivudal statistics in each class.

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        params: parameters of the model given as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']
    """
    if params == None:
        pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
        MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    else:
        pi = params['pi']
        mu_0 = params['mu_0']
        mu_1 = params['mu_1']
        sigma_0 = params['sigma_0']
        sigma_1 = params['sigma_1']
        MLE_phi = params['phi']
        MLE_lambda = params['lambda']

    phi = [1-MLE_phi, MLE_phi]    
    lamb = [1-MLE_lambda, MLE_lambda]    
    mu = [mu_0, mu_1]
    sigma = [sigma_0, sigma_1]

    domain0 = []
    range0 = []
    domain1 = []
    range1 = []

    for (i, j), x_ij in X.items():

        posterior_z = [Z_given_X(X,N,M,params,i,j,0), Z_given_X(X,N,M,params,i,j,1)]

        if posterior_z[1] >= posterior_z[0]:
            domain0.append(x_ij[0, 0])
            range0.append(x_ij[0, 1])
        else:
            domain1.append(x_ij[0, 0])
            range1.append(x_ij[0, 1]) 

    plt.plot(domain1, range1, 'r+')          
    plt.plot(domain0, range0, 'b+')
    p1,  = plt.plot(mu_0[0,0], mu_0[0,1], 'kd')
    p2,  = plt.plot(mu_1[0,0], mu_1[0,1], 'kd')
    plt.show()  

# def X_prob(X,N,M,params, Zi):
#     if params == None:
#         pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
#         MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
#     else:
#         pi = params['pi']
#         mu_0 = params['mu_0']
#         mu_1 = params['mu_1']
#         sigma_0 = params['sigma_0']
#         sigma_1 = params['sigma_1']
#         MLE_phi = params['phi']
#         MLE_lambda = params['lambda']

#     phi = [1.-MLE_phi, MLE_phi]    
#     lamb = [1.-MLE_lambda, MLE_lambda]    
#     mu = [mu_0, mu_1]
#     sigma = [sigma_0, sigma_1]

#     ans = 0.0

#     for y in range(2):
#         p = phi[y]
#         for (i, j), x_ij in X.items():
#             if i != Zi:
#                 continue

#             s = 0.0
#             for z in range(2):
#                 s+=p_xij_given_zij(x_ij, mu[z], sigma[z])*lamb[y==z]
#             p*=s

#         ans += p

#     return ans

# def YZ_given_X(X,N,M,params,Zi,Zj,valY,valZ):
#     if params == None:
#         pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
#         MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
#     else:
#         pi = params['pi']
#         mu_0 = params['mu_0']
#         mu_1 = params['mu_1']
#         sigma_0 = params['sigma_0']
#         sigma_1 = params['sigma_1']
#         MLE_phi = params['phi']
#         MLE_lambda = params['lambda']

#     phi = [1.-MLE_phi, MLE_phi]    
#     lamb = [1.-MLE_lambda, MLE_lambda]    
#     mu = [mu_0, mu_1]
#     sigma = [sigma_0, sigma_1]

#     ans = 0.0

#     p = phi[valY]
#     for (i, j), x_ij in X.items():
#         if i != Zi:
#             continue

#         if j==Zj:
#             p*=p_xij_given_zij(x_ij, mu[valZ], sigma[valZ])*lamb[valY==valZ]
#             continue

#         s = 0.0
#         for z in range(2):
#             s+=p_xij_given_zij(x_ij, mu[z], sigma[z])*lamb[valY==z]
#         p*=s

#     ans += p

#     return ans/X_prob(X,N,M,params, Zi)

# def X_and_Z(X, N, M, params, Zi, Zj, val):
#     if params == None:
#         pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
#         MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
#     else:
#         pi = params['pi']
#         mu_0 = params['mu_0']
#         mu_1 = params['mu_1']
#         sigma_0 = params['sigma_0']
#         sigma_1 = params['sigma_1']
#         MLE_phi = params['phi']
#         MLE_lambda = params['lambda']

#     phi = [1-MLE_phi, MLE_phi]    
#     lamb = [1-MLE_lambda, MLE_lambda]    
#     mu = [mu_0, mu_1]
#     sigma = [sigma_0, sigma_1]

#     ans = 0.0

#     for y in range(2):
#         p = phi[y]
#         for (i, j), x_ij in X.items():
#             if i != Zi:
#                 continue

#             if j==Zj:
#                 p*=p_xij_given_zij(x_ij, mu[val], sigma[val])*lamb[y==val]
#                 continue

#             s = 0.0

#             for z in range(2):
#                 s+=p_xij_given_zij(x_ij, mu[z], sigma[z])*lamb[y==z]
#             p*=s

#         ans += p

#     return ans

# def Z_given_X(X,N,M,params,i,j,val):
#     #xcxc
#     print Y_given_X(X, N, M, params, 0, 1) + Y_given_X(X, N, M, params, 0, 0)
#     print "MATCH"
#     print REAL_Z_given_X(X, N, M, params, 0, 0, 0) 
#     print REAL_Z_given_X(X, N, M, params, 0, 0, 1)
#     print "MATCH"




#    # Z = [X_and_Z(X,N,M,params,i,j,0), X_and_Z(X,N,M,params,i,j,1)]

#     print Y
#     print X_prob(X,N,M,params,i)
#     assert (Z[0] + Z[1]) == X_prob(X,N,M,params,i)

#     return Z[val]/(Z[0] + Z[1])

def perform_em(X, N, M, init_params, max_iters=50, eps=1e-2):
    """Estimate Model B paramters using EM

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        init_params: parameters of the model given as a dictionary
            Dictionary shoudl contain: params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']

    Output:
        params: parameters of the trained model encoded as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']
        log_likelihood: array of log-likelihood values across iterations
    """

    def compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, phi, lambd):
        """Compute the log-likelihood of the data given our parameters

        Input:
            mu_0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
            mu_1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
            sigma_0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
            sigma_1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1
            phi: hyperparameter for princinct preferences
            lambd: hyperparameter for princinct preferences

        Output:
            ll: (float), value of the log-likelihood
        """
        ll = 0.0
        
        # -------------------------------------------------------------------------
        # TODO: Code to compute ll

        for key in X:

            xg0 = p_xij_given_zij(X[key], mu_0, sigma_0)
            xg1 = p_xij_given_zij(X[key], mu_1, sigma_1)

            ll += np.log((1.-phi)*lambd*xg0 + (1.-phi)*(1-lambd)*xg1 + (phi)*(1.-lambd)*xg0 + (phi)*lambd*xg1)
        

        # END_YOUR_CODE
            
        return ll

    mu_0 = init_params['mu_0']
    mu_1 = init_params['mu_1']
    sigma_0 = init_params['sigma_0']
    sigma_1 = init_params['sigma_1']
    phi = init_params['phi']
    lambd = init_params['lambda']

    log_likelihood = [compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, phi, lambd)]
    
    for iter in xrange(max_iters):
        
        # -------------------------------------------------------------------------
        # TODO: Code for the E step

        Z = {}
        Y = {}

        for key in X:
            Z[key] = Z_given_X(X,N,M,init_params,key[0],key[1], 1)
            Y[key] = Y_given_X(X, N, M, init_params, key[0], 1)

        # END_YOUR_CODE

        phi, lambd = 0.0, 0.0
        mu_0 = np.matrix([0.0, 0.0])
        mu_1 = np.matrix([0.0, 0.0])
        sigma_0 = np.matrix([[0.0,0.0],[0.0,0.0]])
        sigma_1 = np.matrix([[0.0,0.0],[0.0,0.0]])


        # -------------------------------------------------------------------------
        # TODO: Code for the M step
        # You need to compute the above variables

        tot = 0.0
        tot0 = 0.0
        tot1 = 0.0
        for key in X:
            phi += Y[key]
            tot += 1.0

            lambd += YZ_given_X(X,N,M,init_params,key[0],key[1],0,0) + YZ_given_X(X,N,M,init_params,key[0],key[1],1,1)

            tot0 += 1.-Z[key]
            tot1 += Z[key]
            mu_0 = np.add(mu_0, (1.-Z[key])*X[key])
            mu_1 = np.add(mu_1, (Z[key])*X[key])

        phi /= tot
        lambd /= tot
        mu_0 /= tot0
        mu_1 /= tot1

        for key in X:
            inside_0 = np.subtract(X[key], mu_0)
            inside_1 = np.subtract(X[key], mu_1)
            sigma_0 += (1.-Z[key]) * np.outer(inside_0, inside_0)
            sigma_1 += Z[key] * np.outer(inside_1, inside_1)
        
        sigma_0/=tot0
        sigma_1/=tot1

        # END_YOUR_CODE

        # Check for convergence
        this_ll = compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, phi, lambd)
        log_likelihood.append(this_ll)
        if len(log_likelihood) > 3 and np.abs((log_likelihood[-2] - log_likelihood[-3]) / log_likelihood[-3]) < eps:
            break

    # pack the parameters and return
    params = {}
    params['pi'] = init_params['pi']
    params['mu_0'] = mu_0
    params['mu_1'] = mu_1
    params['sigma_0'] = sigma_0
    params['sigma_1'] = sigma_1
    params['lambda'] = lambd
    params['phi'] = phi

    print log_likelihood
    print "STOP"

    return params, log_likelihood

#===============================================================================
# This runs the functions that you have defined to produce the answers to the
# assignment problems

#===============================================================================
# pt A.i

#START HERE

pi, mu0, mu1, sigma0, sigma1 = MLE_Estimation()

# colorprint("MLE estimates for PA part A.i:", "teal")
# colorprint("\tpi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
#     %(pi, mu0, mu1, sigma0, sigma1), "red")

# quit()

# #==============================================================================    
# # pt A.ii

# def random_covariance():
#     P = np.matrix(np.random.randn(2,2))
#     D = np.matrix(np.diag(np.random.rand(2) * 0.5 + 1.0))
#     return P*D*P.T

# X, N, M = read_unlabeled_matrix(UNLABELED_FILE)

# # initialization strategy 1
# params = {}
# pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
# MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
# params['pi'] = pi
# params['mu_0'] = mu_0
# params['mu_1'] = mu_1
# params['sigma_0'] = sigma_0
# params['sigma_1'] = sigma_1
# params, log_likelihood = perform_em_modelA(X, N, M, params)
# print log_likelihood
# params_list = [params]
# log_likelihood_list = [log_likelihood]

# for _ in range(2):
#     params = {}
#     params['pi'] = np.random.rand()
#     params['mu_0'] = np.matrix(np.random.randn(1,2))
#     params['mu_1'] = np.matrix(np.random.randn(1,2))
#     params['sigma_0'] = random_covariance()
#     params['sigma_1'] = random_covariance()
#     params, log_likelihood = perform_em_modelA(X, N, M, params)
#     params_list.append(params)
#     log_likelihood_list.append(log_likelihood)

# plt.figure()
# for i, params in enumerate(params_list):
#     print params
#     plt.plot(log_likelihood_list[i])
# plt.legend(['MLE initialization', 'Random initialization', 'Random initialization'], loc=4)
# plt.xlabel('Iteration')
# plt.ylabel('Log likelihood')
# plt.show()

#===============================================================================
# pt. B.i

# MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()

# colorprint("MLE estimates for PA part B.i:", "teal")
# colorprint("\tMLE phi: %s\n\tMLE lambda: %s\n"%(MLE_phi, MLE_lambda), 'red')

# quit()
# #===============================================================================
# # pt B.ii

# X, N, M = read_unlabeled_matrix(UNLABELED_FILE)    
# estimate_leanings_of_precincts(X, N, M, params=None)
# plot_individual_inclinations(X, N, M, params=None)

#===============================================================================
# pt B.iv

def random_covariance():
    P = np.matrix(np.random.randn(2,2))
    D = np.matrix(np.diag(np.random.rand(2) * 0.5 + 1.0))
    return P*D*P.T
X, N, M = read_unlabeled_matrix(UNLABELED_FILE)
# initialization strategy 1
params = {}
pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
params['pi'] = pi
params['mu_0'] = mu_0
params['mu_1'] = mu_1
params['sigma_0'] = sigma_0
params['sigma_1'] = sigma_1
params['phi'] = MLE_phi
params['lambda'] = MLE_lambda
params, log_likelihood = perform_em(X, N, M, params)
print log_likelihood
params_list = [params]
log_likelihood_list = [log_likelihood]

for _ in range(2):
    params = {}
    params['pi'] = np.random.rand()
    params['mu_0'] = np.random.randn(1,2)
    params['mu_1'] = np.random.randn(1,2)
    params['sigma_0'] = random_covariance()
    params['sigma_1'] = random_covariance()
    params['phi'] = np.random.rand()
    params['lambda'] = np.random.rand()
    params, log_likelihood = perform_em(X, N, M, params)
    params_list.append(params)
    log_likelihood_list.append(log_likelihood)

plt.figure()
for i, params in enumerate(params_list):
    print params
    plt.plot(log_likelihood_list[i])
plt.legend(['MLE initialization', 'Random initialization', 'Random initialization'], loc=4)
plt.xlabel('Iteration')
plt.ylabel('Log likelihood')
plt.show()

quit()

#===============================================================================
# pt B.v

X, N, M = read_unlabeled_matrix(UNLABELED_FILE)
# initialization strategy 1
params = {}
pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
params['pi'] = pi
params['mu_0'] = mu_0
params['mu_1'] = mu_1
params['sigma_0'] = sigma_0
params['sigma_1'] = sigma_1
params['phi'] = MLE_phi
params['lambda'] = MLE_lambda
params, log_likelihood = perform_em(X, N, M, params)
for k, v in params.items():
    print k, '=', v
estimate_leanings_of_precincts(X, N, M, params=params)
plot_individual_inclinations(X, N, M, params=params)
