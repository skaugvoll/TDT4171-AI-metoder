import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm


##### ADDED START ####
def lossFunction(w, xn, yn, idx):
    a = logistic_wx(w, xn) - yn
    b = derivation(w, xn, idx)

def derivation(w, xn, idx):
    return xn[idx] * logistic_wx(w,xn) * (1 - logistic_wx(w,xn))

##### ADDED END ####

def logistic_z(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x):
    return logistic_z(np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1
#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features
def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(xrange(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index,:]
        y=y_train[xy_index]
        for i in xrange(dim):
            grad_i = (logistic_wx(w,x) - y) * x[i] * logistic_wx(w,x) * (1-logistic_wx(w,x)) ### TODO DONE : something needs to be done here : implement the derivation found for sigmoid
            w[i] = w[i] - learn_rate * grad_i ### TODO : something needs to be done here : added multiplication between learn_rate and update_grad
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        for i in xrange(dim):
            grad_i = 0.0
            for n in xrange(num_n):
                grad_i += (logistic_wx(w,x) - y) * x[i] * logistic_wx(w,x) * (1-logistic_wx(w,x)) ### TODO DONE:
            w[i] = w[i] - (1 / num_n) * learn_rate * grad_i
    return w

def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
    plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.copper,edgecolors='black')

    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    error=[]
    y_est=[]
    for i in xrange(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm,edgecolors='black')
    print "error=",np.mean(error)
    return w

####
#
# ADDED BY ME
#
####

def print_files(datasets):
    print("Here are the following datasets")
    for index, dataset in enumerate(datasets):
        print("{:<10} - {}".format(index, dataset))

def create_data():
    datasets = [
                "big_nonsep_",
                "big_separable_",
                "small_nonsep",
                "small_separable"
                ]
    print_files(datasets)
    dataset_index = input("so what dataset do you want? ")
    dataset = datasets[int(dataset_index)]

    # Construct training and testing sets
    # the panda DataFrame needs numpy ndarray
    # the numpy hstack stacks arrays in sequence horixontally (collumn wise)
    # and neds a tuple : sequence of ndarrays
    # basicly if we have  examples  = [1,2,3]
    #                     exmp_targ = [y,n,y]
    # hstack gives [ [1, y], [2,n], [3,y] ]


    # np.loadtxt(
    #   file to import as numpy array with shape -->
    #   delimter = \t because all files have a tab between each column
    #   all files have format f1,f2, y --> 2 features thus uses column 0 and 1
    #   and y is column 2
    # )

    # create the training data
    x_train = np.loadtxt("data/data_{}_train.csv".format(dataset), delimiter="\t", usecols=(0,1))
    y_train = np.loadtxt("data/data_{}_train.csv".format(dataset), delimiter="\t", usecols=(2))

    # we also need to create the test data
    x_test = np.loadtxt("data/data_{}_train.csv".format(dataset), delimiter="\t", usecols=(0,1))
    y_test = np.loadtxt("data/data_{}_train.csv".format(dataset), delimiter="\t", usecols=(2))

    print("the shape of x_train is {}\n and y_train is {}".format(x_train.shape, y_train.shape))
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = create_data()

    training_methods = [stochast_train_w, batch_train_w]

    w = train_and_plot(
            xtrain=x_train,
            ytrain=y_train,
            xtest=x_test,
            ytest=y_test,
            training_method=training_methods[0],
            learn_rate=0.1,
            niter=10
        )
