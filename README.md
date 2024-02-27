# BYTE-ML-Task
# importing the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy
dataset = pd.read_csv("Experience_salary.csv")
df = pd.DataFrame(dataset)
df['exp(in months)'] = df['exp(in months)'].round(2)
df['salary(in thousands)'] = df['salary(in thousands)'].round(2)
# data preprocessing
X = dataset.iloc[:, 0].values #independent variable array
y = dataset.iloc[:,-1].values #dependent variable vector
# splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print("Type of x_train:",type(X_train))
print("First five elements of x_train are:\n", X_train[:5].round(2))
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5].round(2))
print ('The shape of x_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))
# Visualization
plt.scatter(X_train, y_train, color = 'red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    pred = w * x + b
    squared_errors = (pred - y) ** 2
    total_cost = (1/(2*m) * np.sum(squared_errors))
    return total_cost
initial_w = 0.9998
initial_b = 1
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print("\n")
print("*********COST FUNCTION**********")
print(type(cost))
print(cost)
def compute_gradient(x, y, w, b):
    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    m = len(x)
    y_pred = w * x + b
    errors = y - y_pred
    # Calculate gradients
    w_gradient = -(1 / m) * np.sum(x * errors)
    b_gradient = -(1 / m) * np.sum(errors)
    return w_gradient, b_gradient
initial_w = 0.998
initial_b = 1
tmp_dj_dw, tmp_dj_db = compute_gradient(X_train, y_train, initial_w, initial_b)
print("\n")
print("*********GRADIENT FUNCTION**********")
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b )
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i<100000:
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
    return w, b, J_history, w_history
# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# gradient descent 
iterations = 2500
alpha = 0.0001
w,b,_,_ = gradient_descent(X_train ,y_train, initial_w, initial_b,compute_cost, compute_gradient, alpha, iterations)
print("\n")
print("*********GRADIENT DESCENT**********")
print("w,b found by gradient descent:", w, b)

m = X_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = w * X_train[i] + b

x_vals = np.linspace(X_test.min(), X_test.max(), 100)
y_pred = w * x_vals + b

# Plot the data and the linear fit for training
plt.scatter(X_train, y_train, label="Training Data")
plt.plot(x_vals, y_pred, color="red", label="Linear Regression Line")

# Customize the plot
plt.xlabel("Experience (months)")
plt.ylabel("Salary")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()

# Plot the data and the linear fit for testing
plt.scatter(X_test, y_test, label="Testing Data")
plt.plot(x_vals, y_pred, color="red", label="Linear Regression Line")

# Customize the plot
plt.xlabel("Experience (months)")
plt.ylabel("Salary")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()

input_values = np.array([17.0, 50.0])
predicted_salaries = w * input_values + b
# predicted salaries
print("\n")
print("******OUTPUT ON GIVEN INPUT VALUES******")
print("Predicted salaries for input values (in months):")
print(predicted_salaries.round(2))


