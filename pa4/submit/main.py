# Gibbs sampling algorithm to denoise an image
# Author : Gunaa AV, Isaac Caswell
# Edits : Bo Wang, Kratarth Goel, Aditya Grover
# Date : 2/17/2017

import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pdb
import time


def markov_blanket(i,j,Y,X):
    '''
    return:
        the a list of Y values that are markov blanket of Y[i][j]
        e.g. if i = j = 1,
            the function should return [Y[0][1], Y[1][0], Y[1][2], Y[2][1], X[1][1]]
    '''
    ########
    # TODO #
    ########
    pass

def sampling_prob(markov_blanket):
    '''
    markov_blanket: a list of the values of a variable's Markov blanket
        The order doesn't matter (see part (a)). e.g. [1,1,-1,1]
    return:
         a real value which is the probability of a variable being 1 given its Markov blanket
    '''
    ########
    # TODO #
    ########
    pass

def sample(i, j, Y, X, DUMB_SAMPLE = 0):
    '''
    return a new sampled value of Y[i][j]
    It should be sampled by
        (i) the probability condition on all the other variables if DUMB_SAMPLE = 0
        (ii) the consensus of Markov blanket if DUMB_SAMPLE = 1
    '''
    blanket = markov_blanket(i,j,Y,X)

    if not DUMB_SAMPLE:
        prob = sampling_prob(blanket)
        if random.random() < prob:
            return 1
        else:
            return -1
    else:
        c_b = Counter(blanket)
        if c_b[1] >= c_b[-1]:
            return 1
        else:
            return -1

def get_posterior_by_sampling(filename, initialization, logfile, DUMB_SAMPLE, burn, iterations, eta, beta, energy=False):
    '''
    Do Gibbs sampling and compute the energy of each assignment for the image specified in filename.
    If not dumb_sample, it should run MAX_BURNS iterations of burn in and then
    MAX_SAMPLES iterations for collecting samples.
    If dumb_sample, run MAX_SAMPLES iterations and returns the final image.

    filename: file name of image in txt
    initialization: 'same' or 'neg' or 'rand'
    logfile: the file name that stores the energy log (will use for plotting later)
        look at the explanation of plot_energy to see detail
    DUMB_SAMPLE: equals 1 if we want to use the trivial reconstruction in part (d)

    return value: posterior, Y, frequencyZ
        posterior: an 2d-array with the same size of Y, the value of each entry should
            be the probability of that being 1 (estimated by the Gibbs sampler)
        Y: The final image (for DUMB_SAMPLE = 1, in part (d))
        frequencyZ: a dictionary with key: count the number of 1's in the Z region
                                      value: frequency of such count
    '''
    print "Read the file"
    X = read_txt_file(filename)
    Y = copy.deepcopy(X)
    
    if initialization == 'same':
      print("Same")      
    elif initialization == 'neg':
      print("Neg")
      for y in range(1,len(Y)-1):
        for x in range(1,len(Y[y])-1):
          Y[y][x] = -Y[y][x]
    elif initialization == 'rand':
      print("Rand")
      for y in range(1,len(Y)-1):
        for x in range(1, len(Y[y])-1):
          Y[y][x] = np.random.randint(0,2)*2-1
    else:
      print "WRONG XCXC"

    posterior = np.zeros(np.array(Y).shape)

    plt.imshow(Y)
    plt.savefig(str(initialization) + "start.png")

    Z={}

    with open(logfile, "w+") as file:

      for i in range(burn):

        if i%100 == 50:
          plt.imshow(Y)
          plt.savefig(str(initialization) + " " + str(i)+".png")

        Y = iter(Y,X,eta,beta)
        if energy:
          energy = calcEnergy(Y,X,eta,beta)
          file.write(str(i+1) + "\t" + str(energy) + "\tB\n")

      for i in range(iterations):

        if i%100 == 0:
          plt.imshow(Y)
          plt.savefig(str(initialization) + " " + str(i)+".png")

        Y = iter(Y,X,eta,beta)
        posterior = posterior + (np.array(Y)==-1)
        if energy:
          energy = calcEnergy(Y,X,eta,beta)
          file.write(str(i+1+burn) + "\t" + str(energy) + "\tS\n")

        count = 0
        for i in range(125,163):
          for j in range(143, 175):
            if Y[i][j] == 1:
              count += 1

        if count in Z:
          Z[count] += 1
        else:
          Z[count] = 0

    posterior = posterior / iterations

    return posterior, Y, Z

def iter(Y,X,eta,beta):
  for y in range(len(Y)-1):
    for x in range(len(Y[y])-1):
      if x == 0 or y == 0:
        continue

      neigh = Y[y+1][x] + Y[y-1][x] +Y[y][x+1] +Y[y][x-1] #maybe optimize here
      prob = 1.0/(1.+ math.exp(-2.*eta*X[y][x]-2.*beta*neigh))

      if np.random.uniform() < prob:
        Y[y][x] = 1
      else:
        Y[y][x] = -1

  return Y

def calcEnergy(Y, X, eta, beta):
  sel = 0.0
  edg = 0.0 # we double count edges

  for y in range(len(Y)-1):
      for x in range(len(Y[y])-1):
        if x == 0 or y == 0:
          continue
        sel += Y[y][x]*X[y][x]
        neigh = Y[y+1][x] + Y[y-1][x] +Y[y][x+1] +Y[y][x-1]
        edg += Y[y][x] * neigh

  energy = -eta*sel-beta*edg/2.

  return energy

def denoise_image(filename, initialization, logfile, DUMB_SAMPLE, iterations, burn, eta, beta):
    '''
    Do Gibbs sampling on the image and return the denoised one and frequencyZ
    '''
    posterior, Y, Z = \
        get_posterior_by_sampling(filename, initialization, logfile, DUMB_SAMPLE, iterations, burn, eta, beta)


    if DUMB_SAMPLE:
        for i in xrange(len(Y)):
            for j in xrange(len(Y[0])):
                Y[i][j] = .5*(1.0-Y[i][j]) # 1, -1 --> 1, 0
        return Y, Z
    else:
        denoised = np.zeros(posterior.shape)
        denoised[np.where(posterior<.5)] = 1
        return denoised,Z


# ===========================================
# Helper functions for plotting etc
# ===========================================

def plot_energy(filename):
    '''
    filename: a file with energy log, each row should have three terms separated by a \t:
        iteration: iteration number
        energy: the energy at this iteration
        S or B: indicates whether it's burning in or a sample
    e.g.
        1   -202086.0   B
        2   -210446.0   S
        ...
    '''
    its_burn, energies_burn = [], []
    its_sample, energies_sample = [], []
    with open(filename, 'r') as f:
        for line in f:
            it, en, phase = line.strip().split()
            if phase == 'B':
                its_burn.append(it)
                energies_burn.append(en)
            elif phase == 'S':
                its_sample.append(it)
                energies_sample.append(en)
            else:
                print "bad phase: -%s-"%phase

    p1, = plt.plot(its_burn, energies_burn, 'r')
    p2, = plt.plot(its_sample, energies_sample, 'b')
    plt.title(filename)
    plt.legend([p1, p2], ["burn in", "sampling"])
    plt.gca().set_ylim([-216000,-212000])
    plt.gca().set_xlim([0,1200])
    plt.savefig(filename)
    plt.close()


def read_txt_file(filename):
    '''
    filename: image filename in txt
    return:   2-d array image
    '''
    f = open(filename, "r")
    lines = f.readlines()
    height = int(lines[0].split()[1].split("=")[1])
    width = int(lines[0].split()[2].split("=")[1])
    Y = [[0]*(width+2) for i in range(height+2)]
    for line in lines[2:]:
        i,j,val = [int(entry) for entry in line.split()]
        Y[i+1][j+1] = val
    return Y


def convert_to_png(denoised_image, title):
    '''
    save array as a png figure with given title.
    '''
    fix = np.array(denoised_image)
    fix[fix==-1] = 0
    fix[fix==0] = 2
    fix[fix==1] = 0
    fix[fix==2] = 1
    denoised_image = fix.tolist()

    plt.imshow(denoised_image, cmap=plt.cm.gray)
    plt.title(title)
    plt.savefig(title + '.png')


def get_error(img_a, img_b):
    '''
    compute the fraction of all pixels that differ between the two input images.
    '''
    N = len(img_b[0])*len(img_b)*1.0
    return sum([sum([1 if img_a[row][col] != img_b[row][col] else 0 for col in           range(len(img_a[0]))])
	 for row in range(len(img_a))]
	 ) /N


#==================================
# doing part (c), (d), (e), (f)
#==================================

def perform_part_c():
    '''
    Run denoise_image function with different initialization and plot out the energy functions.
    '''
    ########
    # TODO #
    ########
    get_posterior_by_sampling('noisy_20.txt', 'same', 'log_same', 0, 100, 1000, 1, 1, True)
    #get_posterior_by_sampling('noisy_20.txt', 'neg', 'log_neg', 0, 100, 1000, 1, 1, True)
    #get_posterior_by_sampling('noisy_20.txt', 'rand', 'log_rand', 0, 100, 1000, 1, 1, True)

    #### plot out the energy functions
    plot_energy("log_rand")
    plot_energy("log_neg")
    plot_energy("log_same")

def perform_part_d():
    '''
    Run denoise_image function with different noise levels of 10% and 20%, and report the errors between denoised images and original image
    '''
    orig_img = read_txt_file('orig.txt')

    denoised_20, _ = denoise_image('noisy_20.txt', 'same', 'log_same', 0, 100, 1000, 1, 1)
    denoised_10, _ = denoise_image('noisy_10.txt', 'same', 'log_same', 0, 100, 1000, 1, 1)
    ########
    # TODO #
    ########

    x = np.array(denoised_10)
    y = np.array(denoised_20)
    z = np.array(orig_img)
    
    z[z==0] = -10
    z[z==-1] = 0

    print("Noise 10 " + str(np.sum(z==x)))
    print("Noise 20 " + str(np.sum(z==y)))
    print("Size = " + str(236 * 360))

    ####save denoised images and original image to png figures
    convert_to_png(denoised_10, "denoised_10")
    convert_to_png(denoised_20, "denoised_20")
    convert_to_png(orig_img, "orig_img")

def dumbAlgo(filename):
  X = read_txt_file(filename)
  Y = copy.deepcopy(X)

  for i in range(50):
    for y in range(len(Y)-1):
      for x in range(len(Y[y])-1):
        if x == 0 or y == 0:
          continue

        neigh = Y[y+1][x] + Y[y-1][x] + Y[y][x+1] + Y[y][x-1] + X[y][x]
        if neigh >= 0:
          Y[y][x] = 1
        else:
          Y[y][x] = -1

  return Y


def perform_part_e():
    '''
    Run denoise_image function using dumb sampling with different noise levels of 10% and 20%.
    '''
    orig_img = read_txt_file('orig.txt')
    denoised_dumb_10 = dumbAlgo("noisy_10.txt")
    denoised_dumb_20 = dumbAlgo("noisy_20.txt")


    ####save denoised images to png figures
    convert_to_png(denoised_dumb_10, "denoised_dumb_10")
    convert_to_png(denoised_dumb_20, "denoised_dumb_20")

    x = np.array(denoised_dumb_10)
    y = np.array(denoised_dumb_20)
    z = np.array(orig_img)
    z[z==0] = -10

    print("Noise 10 " + str(np.sum(z==x)))
    print("Noise 20 " + str(np.sum(z==y)))
    print("Size = " + str(236 * 360))

    print "Part e"

def perform_part_f():
    '''
    Run Z square analysis
    '''

    _, f = denoise_image('noisy_10.txt', 'same', 'log_same', 0, 100, 1000, 1, 1)
    width = 1.0
    plt.clf()
    plt.bar(f.keys(), f.values(), width, color = 'b')
    plt.show()
    _, f = denoise_image('noisy_20.txt', 'same', 'log_same', 0, 100, 1000, 1, 1)
    plt.clf()
    plt.bar(f.keys(), f.values(), width, color = 'b')
    plt.show()

if __name__ == "__main__":
    perform_part_c()
    #perform_part_d()
    #perform_part_e()
    #perform_part_f()
