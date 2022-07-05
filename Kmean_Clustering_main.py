# -- coding: utf-8 --
"""

@author: Mohamed Hisham
"""

# new edition ->>
from mimetypes import init
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import math
import random as rd
import re
import math
import string
import os.path
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def jaccard_Distance(sentence1=[], sentence2=[]):
    
    #   ['word', 'word', 'two']
    #   ['list', 'two', 'word']

    #   ['word', two ]

     sentence1 = set(sentence1)
     
     sentence2 = set(sentence2)
     
     union = len(list((sentence1 | sentence2)))
     
     intersection = len(list((sentence1 & sentence2)))
     return 1-(intersection/union)

# ======================================
# ======================================


def pre_process(data):

    file = open(data, "r", encoding="utf8")
    tweets = list(file)
    list_of_tweets = []

    for tweet in tweets:
        # removing IDs and Timestamps
        tweet = tweet[50:]

        # removing 'new lines'
        tweet = tweet.strip('\n')

        # removing any word starts with '@'
        tweet = re.sub(r" @[a\w.]*", '', tweet)

        # revmoing urls
        tweet = re.sub(r" http:.*$", '', tweet)


        # removing punctuation marks
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))

        # removing any '#' symbol
        tweet = re.sub('#', '', tweet)


        # converting all words to lower case
        tweet = tweet.lower()

        # converting each tweet into a list of strings
        tweet = tweet.split()

        # adding each tweet into a list
        list_of_tweets.append(tweet)

    file.close()
   
    return list_of_tweets


# ======================================
# ======================================


class KMean:
    k = 3
    maxIteration = 5
    clusters = [[], []]
    centroids = [[], []]
    twittDataSet = [[], []]

    # y - axis
    pl_sse = []  # list of sse that appear in the all experiment
    # the inner list belong to size of each cluster, the outer list for each iteration(experiment)
    size_clr = []

    def __init__(self, twittDataSet="bbchealth.txt", k=3, maxIteration=5):
        self.k = k
        self.maxIteration = maxIteration
        self.twittDataSet = twittDataSet
        # list contain k list of clusters [[cluster 1] , [ cluster 2] ,[cluster 3] ],...] (for indecies)
        # define list of k lists with empty values ex: k = 3, list = [ [],[],[] ]
        self.clusters = [[] for x in range(self.k)]
        self.centroids = [[] for x in range(self.k)]

    # --------------------------------

    def Algorithm(self):

        # stroe number of twitts in the data set ( samples )
        self.n_samples = len(self.twittDataSet)

        # pick the centroid(the random point in each cluster in our case we will assign in random index of list in list of lists )
        # random_sample_idxs[ ] in the array will be k random elements(each k is centroid for each cluster)
        # replace = False -> to take different centriods
        random_sample_idxs = np.random.choice(
            self.n_samples, self.k, replace=False)

        # assign centroids
        self.centroids = [self.twittDataSet[idx] for idx in random_sample_idxs]
        self.clusters = self.create_clusters()

        # optmization
        for x in range(self.maxIteration):

            print("\nCalculate the sse for: Iteration", end=" ")
            print((x + 1), " ......\n")
            oldCentroids = self.centroids
            # update centroids
            self.centroids = self.update_centroids()
            # updating cluster
            self.clusters = self.create_clusters()

            self.sse = self.compute_SSE()
            # debug
            print("SSE ->", end="")
            print(self.sse, end="\n\n")

            #  [[1,3,4], [2, 6, 7]]
            self.pl_sse.append(self.sse)
            self.size_clr.append([len(clr)
                                 for clr in self.clusters])  # [2, 6, 7]

            if self.is_converged(oldCentroids, self.centroids):
                break
        return self.sse
    # --------------------------------------------

    def create_clusters(self):
        clusters = [[] for x in range(self.k)]
        for idx, sample in enumerate(self.twittDataSet):
            # store index of the closest cetroid
            centroid_idx = self.closest_centroid(sample)  # 3

            clusters[centroid_idx].append(idx)
        return clusters

    # --------------------------------------------

    def closest_centroid(self, sample):

       
       # we check which list is more similar to which centroid by using jaccard distance
        dissimilarity = [jaccard_Distance(
            sample, self.centroids[idx]) for idx, point in enumerate(self.centroids)]
        # dissimialrity = [ 10, 2, 5, 1 ]
        # we see which list is minimum in less dissimilarity so that's mean that's have more  similarity
        closest_idx = np.argmin(dissimilarity)  # 3
        return closest_idx

    # --------------------------------------------
    
    def is_converged(self, centroids_old, centroids):

        for i in range(self.k):

            if(centroids_old[i] != centroids[i]):
                converge = False
                break

            else:
                converge = True

        return converge

    # --------------------------------------------

    def update_centroids(self):

        newCentroids = []
        for clsr in self.clusters:  # [ ->[2, 3, 4], [1, 5], [7, 8, 9 , 10] ]
            # store the total minimum  distance for each twitt in each cluster
            min = 10000000
            # store the index of the new centroid for each cluster
            centIndx = 0
            # twitt index
            t_Indx = 0
            for t_Indx in clsr:
                totalDistance = 0
                for t_Index2 in clsr:
                    # debug
                    # print(t_Indx, ": ", t_Index2)
                    totalDistance += jaccard_Distance(
                        self.twittDataSet[t_Indx], self.twittDataSet[t_Index2])

                if (totalDistance < min):
                    min = totalDistance
                    centIndx = t_Indx

            newCentroids.append(self.twittDataSet[centIndx])
        return newCentroids

    # --------------------------------------------
      
    # def plot(self):
    #     #  ax.plot(self.pl_sse,   color="red")
    #    # ax.plot(self.size_clr, color="black")
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(self.pl_sse,   'r*-')
    #     plt.plot(self.size_clr, "k*-")

    #     plt.xlabel('Iteration Number')
    #     plt.ylabel('Sum_of_squared_distances')
    #     plt.title('sse and size of each cluster')
    #     plt.show()

    # --------------------------------------------
     
    #compute sum of squard error
    def compute_SSE(self):

        sse = 0
        # clusters = [[3, 2, 6], [], []]
        for i in range(0, self.k):
            for twitt_indx in self.clusters[i]:
                sse += jaccard_Distance(
                    self.twittDataSet[twitt_indx], self.centroids[i])**2

        return sse


##################################

            #############       Taking input     #############
print("-"*70)
print(" "*27,  "KMean Clustering")
print("-"*70)

KUser = int( input(" Type k value ,  (Enter -1 if you wnat to use default value  k=3 ->  "))

print("*"*10) # Don't worry it's just for space 


maxUser = int(input(" Type Max iteration value ,  (Enter -1 if you wnat to use default value Max=5 -> "))

print("*"*10)

# type path user in console without quotes   ---->>   bbchealth.txt
pathUser = input(" Type file path ,  (Enter -1 if you wnat to use default path path=bbchealth.txt  ->")

print("*"*10)
   

             ## check for the validation of the input ##
             
## for (k) value 
if(KUser == -1 or KUser <= 0):
    KUser = 3
    
## for (Max number of iterations) value 
if(maxUser == -1 or maxUser <= 0):
    maxUser = 5

## if the path taken is not valid the program will take the default data path you
## you can change it to with your valid path in your pc
if(os.path.exists(pathUser)):

    data = pathUser

else:
    print(" file will be default value")
    data = "bbchealth.txt"  # <<---- put defalut path here          ##

###################################

#  #   run  #  # 

data = pre_process(data)

# number of experiment how many time k will increase by 1

expirement = 3

# plot_sse = []

# k__ = []


sse = -1
print("------------- expirement ", str((0)), end=" ")
print("-------------\n")

Perform_Kmean = KMean(data, KUser, maxUser)
sse = Perform_Kmean.Algorithm()
# Perform_Kmean.plot()

# plot_sse.append(sse)
# k__.append(KUser)
print("SSE ->", end="")
print(sse, end="\n\n")


KUser += 1
for i in range(1, expirement):

    print("------------- expirement ", str((i )), end=" ")
    print("-------------\n")
    temp = []
    Perform_Kmean = KMean(data, KUser, maxUser)
    sse = Perform_Kmean.Algorithm()
    # temp = Perform_Kmean.Algorithm()
    # plot_sse.append(sse)

    # k__.append(KUser)
    print("SSE ->", end="")
    print(sse, end="\n\n")

    KUser += 1


# ax = plt.subplots()
# ax2 = ax.twinx()

# plot k, sse
# ax.plot(k__, plot_sse,   color="blue")
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(k__, plot_sse, 'r*-')
# plt.xlabel('No of Clusters')
# plt.ylabel('Sum_of_squared_distances')
# plt.title('Sse for each k')
# plt.show()
