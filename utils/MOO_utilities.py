import os
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from cvxopt import matrix, solvers

ORDER = ["SuperFuture", "Apples", "WorldNow", "Electronics123", "Photons", "SpaceNow", "PearPear",
         "PositiveCorrelation", "BetterTechnology", "ABCDE", "EnviroLike", "Moneymakers", "Fuel4",
         "MarsProject", "CPU-XYZ", "RoboticsX", "Lasers", "WaterForce", "SafeAndCare", "BetterTomorrow"]

class DataReader():
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.data = {}

    def read_data(self):
        for filename in os.listdir(self.dir_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.dir_path, filename)
                with open(filepath, "r") as data_file:
                    company_name = data_file.readline().strip()
                    data_length = int(data_file.readline().strip())
                    prices = []
                    while True:
                        line = data_file.readline()
                        if len(line.strip()) == 0:
                            break
                        prices.append(float(line.strip().split()[1]))
                assert len(prices) == data_length
                self.data[company_name] = prices.copy()
    
    def get_data(self):
        return self.data.copy()
    
    def plot(self, company_name=None, val_range=None):
        if company_name is None:
            company_name = random.choice(list(self.data.keys()))

        if company_name not in self.data:
            print(f"Company '{company_name}' not found in data.")
            return

        values = self.data[company_name]
        timesteps = list(range(len(values)))

        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, values, linestyle='-', markersize=3)

        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.title(company_name)

        if val_range is None:
            plt.xticks([0, 101, 201, 301, 401], labels=["T=-1", "T=0", "T=+1", "T=+2", "T=+3"]) 
        else:
            plt.xlim([0, val_range])
            plt.xticks([0, 101, 201, 301, 401], labels=["T=-1", "T=0", "T=+1", "T=+2", "T=+3"]) 

        plt.grid(True)
        plt.show()

    def plot_all(self, val_range=None, figsize=(15, 5), normalized_y=False):
        company_names = list(self.data.keys())
        num_companies = len(company_names)
        cols = 4
        rows = math.ceil(num_companies / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        
        if normalized_y:
            all_values = [value for values in self.data.values() for value in values]
            y_min, y_max = min(all_values), max(all_values)
            margin = (y_max - y_min) * 0.1
            y_min -= margin
            y_max += margin
        
        for i, company_name in enumerate(company_names):
            values = self.data[company_name]
            timesteps = list(range(len(values)))
            
            axes[i].plot(timesteps, values, linestyle='-', markersize=3)
            axes[i].set_title(company_name)
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel("Value")
            axes[i].grid(True)
            
            if val_range is None:
                axes[i].set_xlim([0, 101])
                axes[i].set_xticks([0, 101, 201, 301, 401], labels=["T=-1", "T=0", "T=+1", "T=+2", "T=+3"])
            else:
                axes[i].set_xlim([0, val_range])
                axes[i].set_xticks([0, 101, 201, 301, 401], labels=["T=-1", "T=0", "T=+1", "T=+2", "T=+3"])  
            
            if normalized_y:
                axes[i].set_ylim([y_min, y_max])

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

class RegressionModelsCombined():
    def __init__(self, data, window_size=10, test_size = 0.2):
        self.data = data
        self.data_size = len(data[list(data.keys())[0]])
        self.window_size = window_size
        self.test_size = test_size
        self.models = {}

    def set_windows_size(self, window_size):
        self.window_size = window_size

    def create_X_y(self, num_included_points=101):
        self.X = np.arange(self.data_size-num_included_points, self.data_size).reshape(-1, 1)  # Time points as feature
        self.y = {}
        for asset, prices in self.data.items():
            y = np.array(prices)[len(prices) - num_included_points:]  # Prices as target
            self.y[asset] = y

    def create_X_y_windows(self):
        self.X = {}
        self.y = {}
        self.X_train = {}
        self.y_train = {}
        self.X_test = {}
        self.y_test = {}
        
        for company_name in self.data.keys():
            company_data = self.data[company_name]
            company_X = []
            company_y = []
            
            for i in range(self.window_size, len(company_data)):
                company_X.append(company_data[i - self.window_size:i])
                company_y.append(company_data[i])

            self.X[company_name] = np.array(company_X.copy(), dtype=np.float32)
            self.y[company_name] = np.array(company_y.copy(), dtype=np.float32)
            self.X_train[company_name] = np.array(company_X[:int(len(company_X)*(1-self.test_size))].copy(), dtype=np.float32)
            self.X_test[company_name] = np.array(company_X[int(len(company_X)*(1-self.test_size)):].copy(), dtype=np.float32)
            self.y_train[company_name] = np.array(company_y[:int(len(company_y)*(1-self.test_size))].copy(), dtype=np.float32)
            self.y_test[company_name] = np.array(company_y[int(len(company_y)*(1-self.test_size)):].copy(), dtype=np.float32)
    

    def train_linear(self):
        self.y_preds = {}

        for company_name in self.data.keys():
            model = LinearRegression()
            model.fit(self.X, self.y[company_name])
            self.models[company_name] = model

    def predict_linear(self, num_predictions=100):
        predictions = {}

        X_future = np.arange(self.data_size+1, self.data_size+num_predictions+1).reshape(-1, 1)

        for asset, model in self.models.items():
            y_future = model.predict(X_future)
            predictions[asset] = (X_future.flatten(), y_future)

        return predictions

    def train_linear_partial_windows(self):
        self.y_preds_train = {}
        self.y_preds_test = {}
        for company_name in self.data.keys():
            model = LinearRegression()
            model.fit(self.X_train[company_name], self.y_train[company_name])
            self.models[company_name] = model
            self.y_preds_train[company_name] = self.models[company_name].predict(self.X_train[company_name])
            self.y_preds_test[company_name] = self.models[company_name].predict(self.X_test[company_name])

    def train_linear_full_windows(self):
        self.y_preds_full = {}
        for company_name in self.data.keys():
            model = LinearRegression()
            model.fit(self.X[company_name], self.y[company_name])
            self.models[company_name] = model
            self.y_preds_full[company_name] = self.models[company_name].predict(self.X[company_name])

    def train_sinusoidal():
        pass

    def train_ARIMA():
        pass
    
    def train_XGBOOST():
        pass

    def train_NODE():
        pass
    
    def get_covariance_matrix(self, data_window=1.0):
        arr = np.array([(a:=self.data[k])[int((1-data_window)*len(a)):] for k in ORDER])
        covariance_mat = np.cov(arr)
        
        return covariance_mat


    def plot_predictions(self, predictions, val_range=None, figsize=(15, 5)):
        if self.models is None:
            print("First, train the model before plotting predictions.")
            return

        company_names = list(self.data.keys())
        num_companies = len(company_names)
        cols = 4
        rows = math.ceil(num_companies / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        for i, company_name in enumerate(company_names):
            values = self.data[company_name]
            timesteps = list(range(len(values)))

            axes[i].plot(timesteps, values, linestyle='-', markersize=3, color='blue', label='Actual Values')
            axes[i].plot(predictions[company_name][0], predictions[company_name][1], linestyle='--', markersize=3, label='Predicted Values', color='red')
            
            axes[i].set_title(company_name)
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel("Value")
            axes[i].grid(True)
            axes[i].legend()
            
            if val_range is None:
                axes[i].set_xlim([0, 101])
                axes[i].set_xticks([0, 101, 201, 301, 401], labels=["T=-1", "T=0", "T=+1", "T=+2", "T=+3"])
            else:
                axes[i].set_xlim([0, val_range])
                axes[i].set_xticks([0, 101, 201, 301, 401], labels=["T=-1", "T=0", "T=+1", "T=+2", "T=+3"])  
            
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_test_preds(self, val_range=None, figsize=(15, 5)):
        if self.models is None:
            print("First, train the model before plotting predictions.")
            return

        company_names = list(self.data.keys())
        num_companies = len(company_names)
        cols = 4
        rows = math.ceil(num_companies / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        for i, company_name in enumerate(company_names):
            values = self.data[company_name]
            timesteps = list(range(len(values)))
            test_indices = list(range(len(values) - len(self.y_test[company_name]), len(values)))

            axes[i].plot(timesteps, values, linestyle='-', markersize=3, color='blue', label='Actual Values')
            axes[i].scatter(test_indices, self.y_preds_test[company_name], color='red', label='Predicted Values', marker='x')
            
            axes[i].set_title(company_name)
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel("Value")
            axes[i].grid(True)
            axes[i].legend()
            
            if val_range is None:
                axes[i].set_xticks(range(0, len(values), 10))
            else:
                axes[i].set_xlim([0, val_range])
                axes[i].set_xticks(range(0, val_range, 10))

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_all_preds(self):
        pass

    def get_metrics(self, full=True):
        if self.models is None:
            print("First, initialize the model")
            return
        
        if full and self.y_preds_full is None:
            print("First, initialize the full model")
        
        for company_name in self.data.keys():
            if full:
                print(f"{company_name}:")
                mae = mean_absolute_error(self.y[company_name], self.y_preds_full[company_name])
                mse = mean_squared_error(self.y[company_name], self.y_preds_full[company_name])
                print(f"MAE error: {mae}, MSE error: {mse}")

            else:
                print(f"{company_name}:")
                mae_train = mean_absolute_error(self.y_train[company_name], self.y_preds_train[company_name])
                mse_train = mean_squared_error(self.y_train[company_name], self.y_preds_train[company_name])
                mae_test = mean_absolute_error(self.y_test[company_name], self.y_preds_test[company_name])
                mse_test = mean_squared_error(self.y_test[company_name], self.y_preds_test[company_name])
                print(f"Train MAE error: {mae_train}, MSE error: {mse_train}")
                print(f"Test MAE error: {mae_test}, MSE error: {mse_test}")

class Solver():
    def __init__(self, data, predicted_vals, risks):
        self.data = data
        self.predicted_vals = predicted_vals
        self.risks = risks

        self.current_prices = np.array([x[-1] for x in [self.data[y] for y in ORDER]])
        self.predicted_prices = np.array([x[-1] for x in [self.predicted_vals[y][1] for y in ORDER]])
        self.expected_returns = self.predicted_prices/self.current_prices - 1

    def solve(self, w1, w2):
        n = len(self.expected_returns)

        # Transpose because of library and *2 because of the formula TODO verify
        Q = 2 * w1 * matrix(self.risks.T)
        c = -w2 * matrix(self.expected_returns)

        # Constraints: Sum of weights = 1
        A = matrix(np.ones((1, n)))
        b = matrix(1.0)

        # Inequality constraints: 0 <= w <= 1
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), np.ones(n))))

        sol = solvers.qp(Q, c, G, h, A, b)

        return sol
    
    def solve_wsm(self, step=0.1):
        weights1 = np.arange(0, 1.00001, step)
        weights2 = np.ones(len(weights1)) - weights1

        print(weights1, weights2, weights1 + weights2)
        solutions = []
        for w1, w2 in zip(weights1, weights2):
            print(f"Running for w1={w1}, w2={w2}")
            sol = self.solve(float(w1), float(w2))
            sol_weights = np.array(sol["x"])
            sol_weights = sol_weights / np.sum(sol_weights)
            f1 = sum(sol_weights.reshape(-1) * self.expected_returns)
            f2 = (sol_weights.T @ self.risks @ sol_weights)[0][0]

            solutions.append((f1, f2, sol_weights))

        return solutions    
    
    def plot_front(self, solutions, random_sample=None):
        random_points = []

        if random_sample is not None:
            for _ in range(random_sample):  
                sol_weights = np.random.rand(len(self.expected_returns))
                sol_weights /= np.sum(sol_weights)
                f1 = sum(sol_weights * self.expected_returns)
                f2 = sol_weights.T @ self.risks @ sol_weights

                random_points.append((f1, f2))

        plt.figure(figsize=(10, 9))
        plt.scatter([x[0] for x in random_points], [x[1] for x in random_points], color="blue")
        plt.scatter([x[0] for x in solutions], [x[1] for x in solutions], color="red")

        plt.xlabel("Expected Return")
        plt.ylabel("Risk")
        plt.title("Sampled Decision Variables")
        plt.grid(True)
        plt.show()

    def plot_sampled_decision_variables(self, n=500):
        random_points = []

        for _ in range(n):  
            sol_weights = np.random.rand(len(self.expected_returns))
            sol_weights /= np.sum(sol_weights)
            f1 = sum(sol_weights * self.expected_returns)
            f2 = sol_weights.T @ self.risks @ sol_weights

            random_points.append((f1, f2))
        
        plt.figure(figsize=(10, 9))
        plt.scatter([x[0] for x in random_points], [x[1] for x in random_points])

        plt.xlabel("Expected Return")
        plt.ylabel("Risk")
        plt.title("Sampled Decision Variables")
        plt.grid(True)
        plt.show()
    

    # def plot_predictions(self, company_name=None):
    #     if company_name is None:
    #         company_name = random.choice(list(self.data.keys()))

    #     if company_name not in self.data or company_name not in self.predicted_vals:
    #         print(f"Company '{company_name}' not found in data or predictions.")
    #         return
        
    #     actual_values = self.data[company_name]
    #     predicted_values = self.predicted_vals[company_name][1]
    #     actual_timesteps = list(range(len(actual_values)))
    #     predicted_timesteps = list(range(len(actual_values), len(actual_values) + len(predicted_values)))
        
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(actual_timesteps, actual_values, linestyle='-', markersize=3, label='Actual', color='blue')
    #     plt.plot(predicted_timesteps, predicted_values, linestyle='--', markersize=3, label='Predicted', color='red')
        
    #     plt.xlabel("Timestep")
    #     plt.ylabel("Value")
    #     plt.title(f"Actual vs Predicted - {company_name}")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # def plot_all_predictions(self, val_range=None, figsize=(15, 5), normalized_y=False):
    #     company_names = list(self.data.keys())
    #     num_companies = len(company_names)
    #     cols = 4
    #     rows = math.ceil(num_companies / cols)
        
    #     fig, axes = plt.subplots(rows, cols, figsize=figsize)
    #     axes = axes.flatten()
        
    #     if normalized_y:
    #         all_values = [value for values in self.data.values() for value in values]
    #         all_predictions = [value for values in self.predicted_vals.values() for value in values]
    #         y_min, y_max = min(all_values + all_predictions), max(all_values + all_predictions)
    #         margin = (y_max - y_min) * 0.1
    #         y_min -= margin
    #         y_max += margin
        
    #     for i, company_name in enumerate(company_names):
    #         if company_name not in self.predicted_vals:
    #             continue
            
    #         actual_values = self.data[company_name]
    #         predicted_values = self.predicted_vals[company_name]
    #         actual_timesteps = list(range(len(actual_values)))
    #         predicted_timesteps = list(range(len(actual_values), len(actual_values) + len(predicted_values)))
            
    #         axes[i].plot(actual_timesteps, actual_values, linestyle='-', markersize=3, label='Actual', color='blue')
    #         axes[i].plot(predicted_timesteps, predicted_values, linestyle='--', markersize=3, label='Predicted', color='red')
    #         axes[i].set_title(company_name)
    #         axes[i].set_xlabel("Timestep")
    #         axes[i].set_ylabel("Value")
    #         axes[i].grid(True)
            
    #         if val_range is None:
    #             axes[i].set_xticks(range(0, len(actual_values) + len(predicted_values), 10))
    #         else:
    #             axes[i].set_xlim([0, val_range])
    #             axes[i].set_xticks(range(0, val_range, 10))
            
    #         if normalized_y:
    #             axes[i].set_ylim([y_min, y_max])
            
    #         axes[i].legend()
        
    #     for j in range(i + 1, len(axes)):
    #         fig.delaxes(axes[j])
        
    #     plt.tight_layout()
    #     plt.show()
