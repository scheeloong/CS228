import os
import sys
import numpy as np
from scipy.misc import logsumexp
from collections import Counter
import random
import pdb
import math

# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry
# helpers to learn and traverse the tree over attributes
from tree import get_mst, get_tree_root, get_tree_edges

# pseudocounts for uniform dirichlet prior
alpha = 0.1

def renormalize(cnt):
  '''
  renormalize a Counter()
  '''
  tot = 1. * sum(cnt.values())
  for a_i in cnt:
    cnt[a_i] /= tot
  return cnt

#--------------------------------------------------------------------------
# Naive bayes CPT and classifier
#--------------------------------------------------------------------------
class NBCPT(object):
  '''
  NB Conditional Probability Table (CPT) for a child attribute.  Each child
  has only the class variable as a parent
  '''

  def __init__(self, A_i):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
        - A_i: the index of the child variable
    '''
    raise NotImplementedError()

  def learn(self, A, C):
    '''
    TODO populate any instance variables specified in __init__ to learn
    the parameters for this CPT
        - A: a 2-d numpy array where each row is a sample of assignments 
        - C: a 1-d n-element numpy where the elements correspond to the
          class labels of the rows in A
    '''
    pass

  def get_cond_prob(self, entry, c):
    '''
    TODO return the conditional probability P(X|Pa(X)) for the values
    specified in the example entry and class label c
        - entry: full assignment of variables 
            e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
        - c: the class 
    '''
    pass

class NBClassifier(object):
  '''
  NB classifier class specification
  '''

  num_examples = 0.
  a_ones = None # np array with number of times a_i is 1
  num_y = None # the number of times y is one
  num_0 = None # the number of times y is 0
  a_given_0 = None # P(A_i=1| Y=0)
  a_given_1 = None # P(A_i=1 | Y=1)




  y_on_0 = None # np array with number of times y_i is 1 given a_i is 0
  y_on_1 = None # np array with number of times y_i is 1 given a_i is 1
  limit = math.log(0.5)

  def __init__(self, A_train, C_train):
    self.num_examples = A_train.shape[0]
    num_cols = A_train.shape[1]

    self.a_ones = np.zeros(num_cols)/10
    #self.y_on_0 = np.zeros(num_cols)
    #self.y_on_1 = np.zeros(num_cols)
    self.a_given_0 = np.zeros(num_cols)
    self.a_given_1 = np.zeros(num_cols)/10
    self.num_y = 0.
    self.num_0 = 0.

    for i in range(A_train.shape[0]):
      arr = A_train[i,:]
      y = C_train[i]

      self.a_ones = np.sum([self.a_ones, arr], axis=0)

      if y == 1:
        self.num_y += 1.
        self.a_given_1 = np.sum([self.a_given_1, arr], axis=0)
      else:
        self.num_0 += 1.
        self.a_given_0 = np.sum([self.a_given_0, arr], axis=0)

    #pdb.set_trace()

  def shuffle(self, entry):
    total_prob = 0.0
    missing = np.where(entry==-1)[0]
    s = 0.0

    print "PROB OF EDUC"
    print (self.a_given_1[11] + self.a_given_0[11])/self.num_examples
    return
    for i in range(32):
      prob = 1.0
      for j in range(5):

        rep = missing[j]

        num = 1 if 1<<j & i else 0
        entry[rep] = num
        prob *= (self.a_given[1][entry[self.par[rep]]][rep][num])/ \
          (self.a_given[1][entry[self.par[rep]]][rep][1]+self.a_given[1][entry[self.par[rep]]][rep][0])

      res, log_prob = self.classify(entry,True)

      real_prob = math.exp(log_prob) if res == 1 else 1-math.exp(log_prob)
      s+=prob
      total_prob += real_prob*prob

    return total_prob

  def classify(self, entry):

    total_log_prob = 0.0
    cond_prob = self.num_y/self.num_examples
    bot_neg = 1.- cond_prob

    for i, val in enumerate(entry):
      cond_prob *= (self.a_given_1[i]+alpha)/(16*alpha+self.num_y) if val == 1 else (1.-(self.a_given_1[i]+alpha)/(16*alpha+self.num_y))
      bot_neg *= (self.a_given_0[i]+alpha)/(16*alpha+self.num_0) if val == 1 else (1.-(self.a_given_0[i]+alpha)/(16*alpha+self.num_0))

    total_log_prob += math.log(cond_prob)
    total_log_prob -= math.log(cond_prob + bot_neg)

    #print(total_log_prob)

    if total_log_prob > self.limit:
      return (1, total_log_prob)
    else:
      return (0, math.log(1. - math.exp(total_log_prob)))


    '''
    TODO return the log probabilites for class == 0 and class == 1 as a
    tuple for the given entry
    - entry: full assignment of variables 
    e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1

    NOTE this must return both the predicated label {0,1} for the class
    variable and also the log of the conditional probability of this
    assignment in a tuple, e.g. return (c_pred, logP_c_pred)

    '''
    #return (c_pred, logP_c_pred)


#--------------------------------------------------------------------------
# TANB CPT and classifier
#--------------------------------------------------------------------------
class TANBCPT(object):
  '''
  TANB CPT for a child attribute.  Each child can have one other attribute
  parent (or none in the case of the root), and the class variable as a
  parent
  '''

  def __init__(self, A_i, A_p):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
     - A_i: the index of the child variable
     - A_p: the index of its parent variable (in the Chow-Liu algorithm,
       the learned structure will have a single parent for each child)
    '''
    raise NotImplementedError()

  def learn(self, A, C):
    '''
    TODO populate any instance variables specified in __init__ to learn
    the parameters for this CPT
     - A: a 2-d numpy array where each row is a sample of assignments 
     - C: a 1-d n-element numpy where the elements correspond to the class
       labels of the rows in A
    '''
    pass

  def get_cond_prob(self, entry, c):
    '''
    TODO return the conditional probability P(X|Pa(X)) for the values
    specified in the example entry and class label c  
        - entry: full assignment of variables 
                e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
        - c: the class               
    '''
    pass



class TANBClassifier(NBClassifier):
  '''
  TANB classifier class specification
  '''
  num_examples = 0.
  num_a = 0.
  tree = None
  par = None # par[x] is the non y parent of x. -1 if none
  chil = None # child[x] is list of children of 
  a_given = None # a_given[0][1][x][b] means if y is 0 and parent is 1 number of times node x is b
  num_y_a = None # num_y_a[0][x][0] means number of times y is 0 and a_x is 0
  num_y = None
  topo = None


  def toposort(self):
    vis = [False] * self.num_a

    self.topo = [self.root]
    vis[self.root] = True
    
    for x in range(self.num_a):
      for i in range(self.num_a):
        if not vis[i] and vis[self.par[i]]:
          vis[i] = True
          self.topo.append(i)
          break

  def _learn(self, arr, y, x):
    parent = 0 if self.par[x] == -1 else arr[self.par[x]]
    self.a_given[y][parent][x][arr[x]]+=1.
    #self.num_y_a[y][x][arr[x]]+=1

    for child in self.chil[x]:
      self._learn(arr, y, child)

  def __init__(self, A_train, C_train):

    self.num_examples = A_train.shape[0]
    self.num_a = A_train.shape[1]
    self.tree = get_mst(A_train, C_train)
    self.par = [-1] * self.num_a
    self.chil = [[] for i in range(self.num_a)]
    self.a_given = np.zeros(shape=[2,2,self.num_a,2])
    self.num_y = [0., 0.]

    self.root = get_tree_root(self.tree)
    for dad, kid in get_tree_edges(self.tree, self.root):
      self.par[kid] = dad
      self.chil[dad].append(kid)

    for i in range(A_train.shape[0]):
      arr = A_train[i,:]
      y = C_train[i]

      self._learn(arr, y, self.root)
      self.num_y[y] += 1.

    self.toposort()


    for i in range(16):
      #print "Variable " + str(i)
      #print self.par[i]

      for j in range(2):
        for k in range(2):
          pass
          #print(self.a_given[1][j][i][k]/(self.a_given[1][j][i][0]+self.a_given[1][j][i][1]))

    #print(self.num_y)
    # print(self.num_examples)

    # FIND AN ORDERING TO LOOP THROUGH

    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    _train()
        - A_train: a 2-d numpy array where each row is a sample of
          assignments 
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A

    '''

  def shuffle(self, entry):
    total_prob = 0.0
    missing = np.where(entry==-1)[0]
    s = 0.0

    print "PROB OF EDUC"
    print (self.a_given[1][entry[self.par[11]]][11][1] + self.a_given[0][entry[self.par[11]]][11][1])/ \
          ((self.a_given[1][entry[self.par[11]]][11][1] + self.a_given[0][entry[self.par[11]]][11][1]) + \
          (self.a_given[1][entry[self.par[11]]][11][0] + self.a_given[0][entry[self.par[11]]][11][0]))

    for i in range(32):
      prob = 1.0
      for j in range(5):

        rep = missing[j]

        num = 1 if 1<<j & i else 0
        entry[rep] = num
        prob *= (self.a_given[1][entry[self.par[rep]]][rep][num])/ \
          (self.a_given[1][entry[self.par[rep]]][rep][1]+self.a_given[1][entry[self.par[rep]]][rep][0])

      res, log_prob = self.classify(entry,True)

      real_prob = math.exp(log_prob) if res == 1 else 1-math.exp(log_prob)
      s+=prob
      total_prob += real_prob*prob

    return total_prob

  def classify(self, entry, printer=False):

    total_log_prob = 0.0
    cond_prob = self.num_y[1]/self.num_examples
    bot_neg = 1.- cond_prob
    #print(cond_prob)
    #print(bot_neg)

    for val in range(16):
    #val = 0
      

      parent = self.par[val]
      entry_val = entry[parent] if parent != -1 else 0
      cond_prob *= (self.a_given[1][entry_val][val][entry[val]] + alpha) / \
        (self.a_given[1][entry_val][val][0] + self.a_given[1][entry_val][val][1] + self.num_a*alpha)

      bot_neg   *= (self.a_given[0][entry_val][val][entry[val]] + alpha) / \
        (self.a_given[0][entry_val][val][0] + self.a_given[0][entry_val][val][1] + self.num_a*alpha)

    #print(self.a_given)

    #print(cond_prob)
    #if printer:
      #print(bot_neg)
      #print(cond_prob)
    total_log_prob += math.log(cond_prob)
    total_log_prob -= math.log(cond_prob + bot_neg)

    #pdb.set_trace()

    #print(math.exp(total_log_prob))

    if total_log_prob > self.limit:
      return (1, total_log_prob)
    else:
      return (0, math.log(1. - math.exp(total_log_prob)))

# load all data
A_base, C_base = load_vote_data()

def evaluate(classifier_cls, train_subset=False):
  '''
  evaluate the classifier specified by classifier_cls using 10-fold cross
  validation
  - classifier_cls: either NBClassifier or TANBClassifier
  - train_subset: train the classifier on a smaller subset of the training
    data
  NOTE you do *not* need to modify this function
  
  '''
  global A_base, C_base

  A, C = A_base, C_base

  # score classifier on specified attributes, A, against provided labels,
  # C
  def get_classification_results(classifier, A, C):
    results = []
    pp = []
    for entry, c in zip(A, C):
      c_pred, prob = classifier.classify(entry)
      results.append((c_pred == c))
      pp.append(prob)
    #print 'logprobs', np.array(pp)
    return results
  # partition train and test set for 10 rounds
  M, N = A.shape
  tot_correct = 0
  tot_test = 0
  step = M / 10
  for holdout_round, i in enumerate(xrange(0, M, step)):
    A_train = np.vstack([A[0:i,:], A[i+step:,:]])
    C_train = np.hstack([C[0:i], C[i+step:]])
    A_test = A[i:i+step,:]
    C_test = C[i:i+step]
    if train_subset:
      A_train = A_train[:16,:]
      C_train = C_train[:16]

    # train the classifiers
    classifier = classifier_cls(A_train, C_train)
  
    train_results = get_classification_results(classifier, A_train, C_train)
    #print '  train correct {}/{}'.format(np.sum(nb_results), A_train.shape[0])
    test_results = get_classification_results(classifier, A_test, C_test)
    tot_correct += sum(test_results)
    tot_test += len(test_results)

  return 1.*tot_correct/tot_test, tot_test

def evaluate_incomplete_entry(classifier_cls):

  global A_base, C_base

  # train a TANB classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()




  prob = classifier.shuffle(entry)

  print '  P(C={}|A_observed) = {:2.7f}'.format(1, prob)

  return

def main():
  '''
  TODO modify or add calls to evaluate() to evaluate your implemented
  classifiers
  '''
  
  print 'Naive Bayes'
  accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
  print '  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples)

  print 'TANB Classifier'
  accuracy, num_examples = evaluate(TANBClassifier, train_subset=False)
  print '  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples)

  #print 'Naive Bayes Classifier on missing data'
  #evaluate_incomplete_entry(NBClassifier)

  

  print 'Naive Bayes'
  accuracy, num_examples = evaluate(NBClassifier, train_subset=True)
  print '  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples)

  print 'TANB Classifier'
  accuracy, num_examples = evaluate(TANBClassifier, train_subset=True)
  print '  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples)

  print 'TANB Classifier on missing data'
  evaluate_incomplete_entry(TANBClassifier)

if __name__ == '__main__':
  main()

