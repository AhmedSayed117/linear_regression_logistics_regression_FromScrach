#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas
import numpy as np
import matplotlib.pyplot as plot

# In[2]:
data = pandas.read_csv("car_data.csv")
# drop Column ID
data = data.drop('ID', axis=1)
# shuffle All Rows
data = data.sample(frac=1)
# In[3]:
# d = data.corr()
X = data[["carwidth", "curbweight", "enginesize", "horsepower"]]
y = data[["price"]]

# In[4]:
M = len(data.index)
x_train = X[0:int(M / 2)]
y_train = y[0:int(M / 2)]
x_test = X[int(M / 2):]
y_test = y[int(M / 2):]
x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
alpha = 0.01
theta = np.zeros(5)
theta = theta.reshape((-1, 1))
iterations = 1000

# In[5]:
x_train.insert(0, "X0", 1, True)  # insert only ones

# In[6]:
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()


# In[8]:
def compute_cost(x_train, y_train, theta):
    predictions = x_train.dot(theta)
    errors = np.subtract(predictions, y_train)
    return 1 / (2 * int(M/2)) * errors.T.dot(errors)


# In[9]:
def gradient_descent(x_train, y_train, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = x_train.dot(theta)
        errors = np.subtract(predictions, y_train)
        sum_delta = (alpha / int(M/2)) * x_train.T.dot(errors)
        theta = theta - sum_delta
        cost_history[i] = compute_cost(x_train, y_train, theta)
    return theta, cost_history


# In[10]:
thetaFinal, cost_history = gradient_descent(x_train, y_train, theta, alpha, iterations)

# In[12]:
x = np.arange(1, iterations + 1)
plot.plot(x, cost_history, color='red')
plot.title('MSE Over Each Iterations')
plot.xlabel('Cost For Each Iterations')
plot.ylabel('Iterations')
plot.show()

# In[13]:
print("increment Number Of Iterations += 100 in another 10 Test Cases And alpha Not Changed")
for i in range(20):
    iterations = iterations + 100
    thetaFinal, cost_history = gradient_descent(x_train, y_train, theta, alpha, iterations)
    x = np.arange(1, iterations + 1)
    plot.plot(x, cost_history, color='red')
    plot.title('MSE Over Each Iterations')
    plot.xlabel('Cost For Each Iterations')
    plot.ylabel('Iterations')
    plot.show()

# In[14]:
print("increment alpha += 0.001 in another 20 Test Cases And Iterations Not Changed")
for i in range(20):
    alpha = alpha + 0.001
    thetaFinal, cost_history = gradient_descent(x_train, y_train, theta, alpha, iterations)
    x = np.arange(1, iterations + 1)
    plot.plot(x, cost_history, color='red')
    plot.title('MSE Over Each Iterations')
    plot.xlabel('Cost For Each Iterations')
    plot.ylabel('Iterations')
    plot.show()
