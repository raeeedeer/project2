import math
import copy
import sys
import numpy as np
import time 
import pandas as pd
import csv
#small 102
#large 45
#(16:39)
#(16:23)
#for entire program I referenced Dr. Keogh's Skeleton code and slides on Project 2 breifing 
#https://www.dropbox.com/sh/rltooq0t3khobuj/AAA3MYkZc8gb1RLa3tNSnsrga?dl=0&preview=Project_2_Briefing.pptx

def main():
    print("Raeed Shaikh\"s Feature Selection Algortihm")


    file = input('\nfile name to test: ')
    fn = file
    file = open(file, 'r')

    # Reference for reading the file: https://docs.python.org/3/library/csv.html
    s = csv.reader(file, delimiter=' ', skipinitialspace=True)
    nFeatures = len(next(s))
    
    algoSelector(fn,nFeatures)

def algoSelector(fn,nFeatures):
    print("1) Forward Selection\n")
    print("2) Backward Elimination\n")
    algo = input("Enter the number of the algorithm you want to run.\n")

    if algo not in {'1', '2'}:
        return("Invalid algorithm input, Exiting.")
    if algo == '1':
        start = time.time()
        df = pd.read_fwf(fn, header=None)
        # Reading in file into dataframe:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_fwf.html
        forwardSearch(df,nFeatures)
        print('Time used: ' + str(round(time.time()-start, 2)) + 'seconds.')
    elif algo == '2':
        start = time.time()
        # Reading in file into dataframe:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_fwf.html
        df = pd.read_fwf(fn, header=None)
        backwardElimination(df,nFeatures)
        print('Time used: ' + str(round(time.time()-start, 2)) + ' seconds.')

def getResult(featureD):
    print('Finished search, best feature subset is ' + str(featureD[max(featureD.keys())]) +
        ' with accuracy of ' + "{:.1%}".format(max(featureD.keys())) + '\n')

def accSetter(currAcc,Maxacc):
    if currAcc >= Maxacc:
        Maxacc = currAcc
        return Maxacc

# def colEdit(seenFeatures, finalCol,featureAcc,fd,algo):
#     if algo == '1':
#         seenFeatures.add(finalCol)    
#         s_c = copy.deepcopy(seenFeatures)
#         fd[featureAcc] = s_c
#     elif algo == '2':
#         seenFeatures.remove(finalCol)
#         s_c = copy.deepcopy(seenFeatures)
#         fd[featureAcc] = s_c

def distCalc(distance,nDist):
    if distance <= nDist:
        nDist = distance
    return nDist
       
def forwardSearch(data, Fnum):
    seenFeatures = set()
    fd = {}

    for i in range(1, Fnum):
        maxaccuracy = 0
        finalj = 0
        for j in range(1, Fnum):
            if j not in seenFeatures:

                # deep copy since python does not support ref calls 
                s_temp = copy.deepcopy(seenFeatures)
                s_temp.add(j)


                # Deep copy the DF
                FTA = data.copy(deep=True)[:-1]
                accuracy = leave_one_out_cross_validation(Fnum, s_temp, FTA)
                j_temp = j

                print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(accuracy))

                #update accuracy if new higher accuracy is found
                if accuracy >= maxaccuracy:
                    maxaccuracy = accSetter(accuracy,maxaccuracy)
                    f_accuracy = accuracy
                    finalj = j_temp

        # Add best column 
        seenFeatures.add(finalj)
        s_c = copy.deepcopy(seenFeatures)
        fd[f_accuracy] = s_c

        # float as percent:
        print('Feature set ' + str(seenFeatures) + ' was best, accuracy is ' + "{:.1%}".format(f_accuracy) + '\n')

        getResult(fd)

def backwardElimination(data, Fnum):
    seen_features = set()
    for j in range(1, Fnum):
        seen_features.add(j)
    fd = {}

    #first full iteration must be done outside of loop
    s_temp = copy.deepcopy(seen_features)

    FTA = data.copy(deep=True)
    accuracy = leave_one_out_cross_validation(Fnum, s_temp, FTA)

    print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(accuracy))
    print('Feature set ' + str(s_temp) + ' was best, accuracy is ' + "{:.1%}".format(accuracy) + '\n')

    for i in range(2, Fnum):
        maxaccuracy = 0
        finalj = 0

        for j in range(1, Fnum):
            if j in seen_features:
                s_temp = copy.deepcopy(seen_features)

                # Temporarily remove 
                s_temp.remove(j)

               
                # Deep copy df so its updated in function as well
                FTA = data.copy(deep=True)

                accuracy = leave_one_out_cross_validation(Fnum, s_temp, FTA)
                j_temp = j

                # Printing float 
                print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(accuracy))

                #update accuracy if new higher accuracy is found 
                if accuracy >= maxaccuracy:
                    maxaccuracy = accSetter(accuracy,maxaccuracy)
                    f_accuracy = accuracy
                    #update index
                    finalj = j_temp

        # Remove best column 
        seen_features.remove(finalj)
        s_c = copy.deepcopy(seen_features)
        fd[f_accuracy] = s_c

        # Printing float percent:
        print('Feature set ' + str(seen_features) + ' was best, accuracy is ' + "{:.1%}".format(f_accuracy) + '\n')
        print('\n')

        getResult(fd)

def leave_one_out_cross_validation(Data, seen, FTA):
    nr = len(FTA.index)
    numClassified = 0

    data = FTA.copy(deep=True)

    # convert df to numpy array
    narr = data.to_numpy()
    FTA = narr

    # set col to 0 if not in seen
    for i in range(1, Data):
        if i not in seen:
            FTA[:, i] = 0.0

    # Loop through the array and compute the distance
    for k in range(nr):
        # subsection and corresponding label
        objectClassifier = FTA[k][1:Data]
        LOC = FTA[k][0]

        nDist,nLoc = sys.maxsize,sys.maxsize

        for l in range(nr):
            dist = 0

            # update k if distance is closer 
            if k != l:
                fd = {}

                # add arrays in one line 
                dist = math.sqrt(np.sum(np.power(objectClassifier - FTA[l][1:Data], 2)))
                # calc distance and update if closer
                if dist <= nDist:
                    nDist = distCalc(dist,nDist)
                    nLoc = l + 1
                    nlabel = FTA[nLoc - 1][0]

        # correct classify means increment 
        if LOC == nlabel:
            numClassified += 1
    accuracy = numClassified/nr
    #return updated accuracy
    return accuracy


if __name__ == "__main__":
    main()



    
    
