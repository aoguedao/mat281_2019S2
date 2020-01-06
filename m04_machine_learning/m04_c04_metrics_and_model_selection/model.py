import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import random

secret_true_params = (5., np.pi/4.0, 0., 0.)

def my_model(x, a, b, c, d):
    return a*np.cos(b*x+c) + d

def generate_data(N=100, true_params=secret_true_params,
                  seed = 42):
    x = np.linspace(-2.5, 2.5, N)
    y1 = my_model(x, *true_params)
    y2 = 1.0 * random.normal(size=N)
    # Create the data
    data = np.array([x,y1+y2]).T
    # Shuffle the data
    permuted_data = random.permutation(data)
    # Save the data
    np.savetxt("dataN%d.txt"%N, data)
    return data

def load_data(myfile):
    data = np.loadtxt(myfile)
    return data

def get_params(data):
    # Use optimize to get A and B using the data
    xdata = data[:,0]
    ydata = data[:,1]
    popt, pcov = optimize.curve_fit(my_model, xdata, ydata, maxfev=5000)
    return popt

def get_error(model_params, data):
    x_data = data[:,0]
    y_data = data[:,1]
    y_prediction = my_model(x_data, *model_params)
    #error_1 = np.abs(y_data-y_prediction).sum() / len(y_data)
    error_2 = np.sum((y_data-y_prediction)**2).sum()**0.5 / len(y_data)**0.5
    return error_2

def plot(training_data, testing_data, training_params, all_data_params, true_params=secret_true_params):
    fig = plt.figure(figsize=(16,8))
    plt.plot(training_data[:,0], training_data[:,1], 'bs', label="training data", alpha=0.75, ms=10)
    plt.plot(testing_data[:,0], testing_data[:,1], 'ro', label="testing data", alpha=0.75, ms=10)
    data = np.vstack([training_data, testing_data])
    x = np.array(sorted(data[:,0].copy()))
    plt.plot(x, my_model(x, *true_params),
           'k', label="true params", lw=2.0)
    plt.plot(x, my_model(x, *training_params),
           'b', label="training params", lw=2.0)
    plt.plot(x, my_model(x, *all_data_params),
             'g', label="all data params", lw=2.0)
    xmin, xmax = x.min(), x.max()
    plt.xlim([xmin-.2, xmax+0.2])
    plt.legend(numpoints=1, loc="lower center")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return

def full_report(training_data, testing_data, training_params, all_data_params):
    data = np.vstack([training_data, testing_data])
    print("The obtained model parameters for training dataset are:")
    print("\t(a,b,c,d) = (%.3f, %.3f, %.3f, %.3f)" %tuple(training_params))
    print("The obtained model parameters for the whole dataset are:")
    print("\t(a,b,c,d) = (%.3f, %.3f, %.3f, %.3f)" %tuple(all_data_params))
    print("The true model parameters are:")
    print("\t(a,b,c,d) = (%.3f, %.3f, %.3f, %.3f)" %tuple(secret_true_params))

    print("")
    prediction_error = get_error(training_params, testing_data)
    print("Conservative error estimation on testing dataset: %.2f" %prediction_error)
    true_error = get_error(secret_true_params, testing_data)
    print("Pure random error on testing dataset: %.2f" %true_error)
    all_error = get_error(secret_true_params, data)
    print("Pure random error on all data: %.2f" %all_error)


if __name__=="__main__":
    generate_data(N=20)
    generate_data(N=50)
    generate_data(N=100)
    generate_data(N=500)
    generate_data(N=5000)
