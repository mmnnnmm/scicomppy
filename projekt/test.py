import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import random as random
import numpy as np
import math


# activation functions

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))

def tanh(z):
    return math.tanh(z)

def tanh_prime(z):
    return 1.0-tanh(z)**2

def relu(z):
    return max(0.0,z)

def relu_prime(z):
    return 1.0 if z > 0 else 0.0

def leaky_relu(z):
    return max(0.1*z,z)

def leaky_relu_prime(z):
    return 1.0 if z > 0 else 0.1


# random products
def generate_id(dim,l,sd):
    a = np.ones(dim)/math.sqrt(dim)  
    for i in range (l):    
        x = np.random.normal(0,sd,(dim,dim))
        a = np.dot(x,a)
        
    a = a**2
    return math.log(math.sqrt(sum(i for i in a)))

def generate_tanh(dim,l,sd):
    a = np.ones(dim)/math.sqrt(dim)     
    for i in range (l):    
        x = np.random.normal(0,sd,(dim,dim))
        a = np.tanh(np.dot(x,a))
    a = a**2
    return math.log(math.sqrt(sum(i for i in a)))

def generate_leaky(dim,l,sd):
    a = np.ones(dim)/math.sqrt(dim)      
    for i in range (l):    
        x = np.random.normal(0,sd,(dim,dim))
        a = np.vectorize(leaky_relu)(np.dot(x,a))
    a = a**2
    return math.log(math.sqrt(sum(i for i in a)))

def generate_relu(dim,l,sd):
    a = np.ones(dim)/math.sqrt(dim)  
    for i in range (l):    
        x = np.random.normal(0,sd,(dim,dim))
        a = np.vectorize(relu)(np.dot(x,a))  + 20*l*np.ones(dim)/(0.01*math.sqrt(dim))
    a = a**2
    return math.log(math.sqrt(max(0.1,sum(i for i in a))))

def generate_sigmoid(dim,l,sd):
    a = np.ones(dim)/math.sqrt(dim)  
    for i in range (l):    
        x = np.random.normal(0,sd,(dim,dim))
        a = sigmoid(np.dot(x,a))
    a = a**2
    return math.log(math.sqrt(sum(i for i in a)))

# A function to plot distributions of random products
def histograms():
    n = 10000
    a = [generate_id(4,10,0.55) for i in range (n)]
    b = [generate_tanh(4,10,1) for i in range (n)]
    c = [generate_sigmoid(10,10,1) for i in range (n)]
    d = [generate_leaky(4,10,1) for i in range (n)]
    e = [generate_relu(4,50,1) for i in range (n)]
    
    
    plt.figure(1)
    plt.hist(a,100)
    plt.title("normal")
    
    plt.figure(2)
    plt.hist(b,100)
    plt.title("tanh")
    
    plt.figure(3)
    plt.hist(c,100)
    plt.title("sigmoid")
    
    plt.figure(4)
    plt.hist(d,100)
    plt.title("leaky relu")
    
    plt.figure(5)
    plt.hist(e,100)
    plt.title("relu")

def function (function):
    
    
    if function == "sigmoid" :
        f = sigmoid
        f_prime = sigmoid_prime
    elif function == "tanh":
        f = tanh
        f_prime = tanh_prime
    elif function == "leaky":
        f = leaky_relu
        f_prime = leaky_relu_prime
    elif function =="identity":
        f = lambda y: y
        f_prime = lambda y: 1.0    
    else:
        f = relu
        f_prime = relu_prime
    return [f,  f_prime]


sd_3 = {"identity":0.21, "tanh":0.4, "relu":0.1, "leaky":0.1, "sigmoid":1}
sd_10={"identity":0.21, "tanh":0.35, "relu":0.3, "leaky":0.3, "sigmoid":0.5}


# Lyapunov initialization
def initialization(activation, samples, sizes):
    length = len(sizes)
    s = sd_10[activation]
    f = function(activation)[0]
    
    out = [np.random.randn(sizes[1], sizes[0])]
    test = []
    probne= [np.random.randn(sizes[0],1) for i in range(1)]
    if samples == 1: 
        
        for i in range (length-2):
            out.append(np.random.normal(0, s, (sizes[1], sizes[1])))
        score = 0
        for j in range(len(probne)):
            w = probne[j]
            w = np.vectorize(f)(np.dot(out[0],w))
            
            
            
            score += (math.sqrt(sum(i[0]**2 for i in w)))
        score = score/1
        print(abs(score-1 ))
        
    else:
        
        minimal = 0.3
        for i in range (samples):
            
            x = [np.random.normal(0, s, (sizes[1], sizes[1])) for i in range (length-2)]
            score_x = 0
            if i ==0:
                curr = x
            for j in range(len(probne)):
                w = probne[j]
                w = np.vectorize(f)(np.dot(out[0],w))
                for i in range (length-2):
                    w = np.vectorize(f)(np.dot(x[i],w))
                
                
                score_x += (math.sqrt(sum(i[0]**2 for i in w)))
            score_x = score_x/1          
            #test.append(math.log(max(0.01,min(score_x,100))))
            if minimal == None or abs(score_x-1 )< minimal:
                minimal = abs(score_x-1 )
                curr = x
                
        for i in curr:
            out.append(i)
        print(minimal)
        #plt.figure(2)
        #plt.hist( test,100)
        
        v = probne[0]
        for i in out:
            v = np.vectorize(f)(np.dot(i, v))
        
    return out

    

    
