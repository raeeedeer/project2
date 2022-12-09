import math
import copy
import sys
import numpy 
import time 
import pandas  
import csv

def filereader(filename):
    in_f = filename
    filename = open(filename,'r')
    r = csv.reader(filename, delimiter=' ', skipinitialspace=True)
    r1 = len(next(r))


def main():
    print("Welcome to Raeed Shaikh\"s Feature Selection Algortihm")

    f  = input("name of file to test")
    filereader(f)
    print("\n")

    print("Enter the number of the algorithm you want to run.\n")
    print("1) Forward Selection\n")
    print("2) Backward Elimination\n")
    algorithm = input().strip()
    if algorithm not in {"1", "2"}:
        print("Invalid algorithm input, Exiting.")

    
    
