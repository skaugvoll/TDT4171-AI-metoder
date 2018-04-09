import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import time


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
                x = x_train[n] # get the training example features
                y = y_train[n] # get the training example target
                grad_i += (logistic_wx(w,x) - y) * x[i] * logistic_wx(w,x) * (1-logistic_wx(w,x)) ### TODO DONE:
            w[i] = w[i] - learn_rate * grad_i/num_n
    return w

def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10, data_name="", plot=False, save=False):
    plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.copper,edgecolors='black')

    #train weights
    start = time.time()
    w=training_method(xtrain,ytrain,learn_rate,niter)
    end = time.time()
    time_used = end-start
    error=[]
    y_est=[]
    for i in xrange(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm,edgecolors='black')
    final_error = np.mean(error)
    print "error=",final_error
    plt.title("DS: {}, #IT: {}, TM: {}".format(data_name, niter, training_method.__name__))
    if plot:
        plt.draw()
        if save:
            plt.savefig('figs/plot_{}.png'.format(niter))
    return w, final_error, time_used

####
#
# ADDED BY ME
#
####

def print_files(datasets):
    print("Here are the following datasets")
    for index, dataset in enumerate(datasets):
        print("{:<10} - {}".format(index, dataset))

def create_data(datasets, bypass=False):
    if bypass:
        print_files(datasets)
        dataset_index = input("so what dataset do you want? ")
        dataset = datasets[int(dataset_index)]
    else:
        dataset = datasets

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

    # print("the shape of x_train is {}\n and y_train is {}".format(x_train.shape, y_train.shape))
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    '''
    Keep track of training time, averge error (gets from train_and_plot)
    '''
    datasets = [
                "big_nonsep",
                "big_separable",
                "small_nonsep",
                "small_separable"
                ]

    training_methods = [
        stochast_train_w,
        batch_train_w
        ]

    #### TESTING START
    # x_train, y_train, x_test, y_test = create_data(datasets, bypass=True)
    #
    # w, error, time_used = train_and_plot(
    #         xtrain=x_train,
    #         ytrain=y_train,
    #         xtest=x_test,
    #         ytest=y_test,
    #         training_method=training_methods[0],
    #         learn_rate=0.1,
    #         niter=10
    #     )
    #### TESTING END


    '''
    For each dataset train a Perceptron using both batch and stochastic gradient descent.
    Keep track of the following:
        training time,
        average error (number of errors over total number of testing samples)
    '''
    # print("Each dataset, both methods")
    # tracker = []
    # lr = 0.1
    # niter = 1
    #
    # for d in datasets:
    #     x_train, y_train, x_test, y_test = create_data(d, bypass=False)
    #     for method in training_methods:
    #         w, error, time_used  = train_and_plot(
    #                 xtrain=x_train,
    #                 ytrain=y_train,
    #                 xtest=x_test,
    #                 ytest=y_test,
    #                 training_method=method,
    #                 learn_rate=lr,
    #                 niter=niter,
    #                 data_name=d,
    #                 plot=False
    #             )
    #         tracker.append(
    #             (d, method.__name__, error, time_used)
    #         )
    #
    # print("\n\n{:<25} {:<22} {:>10}% {:>10}".format("Dataset", "Method", "Error","Time used"))
    # for t in tracker:
    #     print("{:<25} {:<22} {:>10} {:>10.2f}s".format(t[0], t[1], t[2]*100, t[3]))




    '''
    Choose one dataset and make a
    * scatter plot the training points, coloring each class with a different color.
    Now
    * run your trained Perceptron over the testing dataset and
      * plot in the same graph
    the testing point coloring each class your diferent colors (use different color for training and testing).
    This way you should visualize for good your model
    '''
    # I choose stochastic on small small_separable dataset
    print("One dataset, one training algorithm, plotting training data than plotting testing in same fig")
    tracker = []
    lr = 0.1
    niter = 100
    d = datasets[3] # small_separable
    x_train, y_train, x_test, y_test = create_data(d, bypass=False)

    w, error, time_used  = train_and_plot(
            xtrain=x_train,
            ytrain=y_train,
            xtest=x_test,
            ytest=y_test,
            training_method=training_methods[0],
            learn_rate=lr,
            niter=niter,
            data_name=d,
            plot=True,
            save=False
        )
    tracker.append(
        (d, niter, training_methods[0].__name__, error, time_used)
    )

    print("\n\n{:<25} {:<10} {:<22} {:>10}% {:>10}".format("Dataset", "iterations", "Method", "Error","Time used"))
    for t in tracker:
        print("{:<25} {:<10} {:<22} {:>10} {:>10.2f}s".format(t[0], t[1], t[2], t[3]*100, t[4]))
        # print(t)



    '''
    * Choose one dataset and one training algorithm (batch or stochastic)
    and run multiple times training and testing varying the number of iterations
    (you should at least try to following number of iterations T = 10, 20, 50, 100, 200, 500),
    keeping track of the running time and average error for each.
    * Plot the results (training time, error, iterations)
    '''
    # print("One dataset, one training algorithm, different iterations")
    # 
    # tracker = []
    # lr = 0.1
    # # niters = [10,20]
    # niters = [10,20,50,100,200,500]
    # d = datasets[3]
    # x_train, y_train, x_test, y_test = create_data(d, bypass=False)
    #
    # for niter in niters:
    #     w, error, time_used  = train_and_plot(
    #             xtrain=x_train,
    #             ytrain=y_train,
    #             xtest=x_test,
    #             ytest=y_test,
    #             training_method=training_methods[0],
    #             learn_rate=lr,
    #             niter=niter,
    #             data_name=d,
    #             plot=True,
    #             save=True
    #         )
    #     tracker.append(
    #         (d, niter, training_methods[0].__name__, error, time_used)
    #     )
    #
    # print("\n\n{:<25} {:<10} {:<22} {:>10}% {:>10}".format("Dataset", "iterations", "Method", "Error","Time used"))
    # for t in tracker:
    #     print("{:<25} {:<10} {:<22} {:>10} {:>10.2f}s".format(t[0], t[1], t[2], t[3]*100, t[4]))
    #     # print(t)

    plt.show()
