import os
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from cvxopt import matrix, solvers
import xgboost as xgb
import torch
import torch.nn as nn
import torchdiffeq

ORDER = ["SuperFuture", "Apples", "WorldNow", "Electronics123", "Photons", "SpaceNow", "PearPear",
         "PositiveCorrelation", "BetterTechnology", "ABCDE", "EnviroLike", "Moneymakers", "Fuel4",
         "MarsProject", "CPU-XYZ", "RoboticsX", "Lasers", "WaterForce", "SafeAndCare", "BetterTomorrow"]

def sinusoidal_function(x, A1, B1, C1, A2, B2, C2, D):
    return A1 * np.sin(B1 * x + C1) + A2 * np.sin(B2 * x + C2) + D

def complex_sinusoidal_function(x, a, A1, B1, C1, A2, B2, C2, D):
    return a * x + A1 * np.sin(B1 * x + C1) + A2 * np.sin(B2 * x + C2) + D

def flat_append(lst1, lst2):
    if isinstance(lst1, np.ndarray): lst1 = lst1.tolist()
    if isinstance(lst2, np.ndarray): lst2 = lst2.tolist()
    result = []
    for val in lst1:
        result.append(val)
    for val in lst2:
        result.append(val)
    return result

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_dim)
        )

    def forward(self, t, y):
        return self.net(y)

class NODETimeSeries(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NODETimeSeries, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        self.ode_solver = torchdiffeq.odeint
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        h0 = self.encoder(x)
        hT = self.ode_solver(self.odefunc, h0, t)
        return self.decoder(hT)

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
    
    def plot(self, company_name=None):
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
        plt.xticks([0, 101, 201, 301, 401], labels=["T=-1", "T=0", "T=+1", "T=+2", "T=+3"]) 

        plt.grid(True)
        plt.show()

    def plot_all(self, figsize=(15, 5), normalized_y=False):
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
            axes[i].set_xticks([0, 101, 201, 301, 401], labels=["T=-1", "T=0", "T=+1", "T=+2", "T=+3"]) 
            
            if normalized_y:
                axes[i].set_ylim([y_min, y_max])

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

class RegressionModelsCombined():
    def __init__(self, data, window_size=10):
        self.data = data
        self.data_size = len(data[list(data.keys())[0]])
        self.window_size = window_size
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
        
        for company_name in self.data.keys():
            company_data = self.data[company_name]
            company_X = []
            company_y = []
            
            for i in range(self.window_size, len(company_data)):
                company_X.append(company_data[i - self.window_size:i])
                company_y.append(company_data[i])

            self.X[company_name] = np.array(company_X.copy(), dtype=np.float32)
            self.y[company_name] = np.array(company_y.copy(), dtype=np.float32)
    
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

    def train_sinusoidal(self): #I am sinusoidal, I am gonna curve myself
        self.y_preds = {}
        for company_name in self.data.keys():
            y = self.data[company_name]
            p0 = [3, 1, 0, 2, 2, 0, np.mean(y)]
            X = np.squeeze(self.X)
            X_scaled = (X - np.min(X)) / (np.max(X) - np.min(X)) * (2 * np.pi)
            params, _ = curve_fit(sinusoidal_function, X_scaled, y, p0=p0, maxfev=500000)
            self.models[company_name] = params
            self.y_preds[company_name] = sinusoidal_function(np.squeeze(X_scaled), *params)

    def predict_sinusoidal(self, num_predictions=100):
        predictions = {}
        X_future = np.arange(self.data_size + 1, self.data_size + num_predictions + 1)
        X_future_scaled = (X_future - np.min(self.X)) / (np.max(self.X) - np.min(self.X)) * (2 * np.pi)
        for asset, params in self.models.items():
            y_future = sinusoidal_function(X_future_scaled, *params)
            predictions[asset] = (X_future, y_future)
        return predictions
    
    def train_complex_sinusoidal(self):
        self.y_preds = {}
        for company_name in self.data.keys():
            y = self.data[company_name]
            p0 = [1, 3, 1, 0, 2, 2, 0, np.mean(y)]
            X = np.squeeze(self.X)
            X_scaled = (X - np.min(X)) / (np.max(X) - np.min(X)) * (2 * np.pi)
            params, _ = curve_fit(complex_sinusoidal_function, X_scaled, y, p0=p0, maxfev=500000)
            self.models[company_name] = params
            self.y_preds[company_name] = complex_sinusoidal_function(np.squeeze(X_scaled), *params)

    def predict_complex_sinusoidal(self, num_predictions=100):
        predictions = {}
        X_future = np.arange(self.data_size + 1, self.data_size + num_predictions + 1)
        X_future_scaled = (X_future - np.min(self.X)) / (np.max(self.X) - np.min(self.X)) * (2 * np.pi)
        for asset, params in self.models.items():
            y_future = complex_sinusoidal_function(X_future_scaled, *params)
            predictions[asset] = (X_future, y_future)
        return predictions
    
    def plot_arima_lags(self, max_lags=20):
        for company_name in self.data.keys():
            y = self.data[company_name]
            print(f"Lags for company = {company_name}")
            plot_acf(y, lags=max_lags)
            plot_pacf(y, lags=max_lags)
            plt.show()

    def train_arima(self, p, d, q):
        self.models = {}
        for company_name in self.data.keys():
            y = self.data[company_name]
            model = ARIMA(y, order=(p, d, q))
            fitted_model = model.fit()
            self.models[company_name] = fitted_model

    def predict_arima(self, num_predictions=100):
        predictions = {}
        X_future = np.arange(self.data_size + 1, self.data_size + num_predictions + 1).reshape(-1, 1)
        for company_name, model in self.models.items():
            forecast = model.forecast(steps=num_predictions)
            predictions[company_name] = (X_future.flatten(), forecast)
        return predictions
    
    def train_xgboost(self):
        self.models = {}
        for company_name in self.data.keys():
            X_train = self.X[company_name]
            y_train = self.y[company_name]
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4)
            model.fit(X_train, y_train)
            self.models[company_name] = model
        
    def predict_xgboost(self, num_predictions=100):
        predictions = {}
        X_future = np.arange(self.data_size + 1, self.data_size + num_predictions + 1).reshape(-1, 1)
        for company_name in self.data.keys():
            curr_num_preds = 0
            input_data = np.expand_dims(np.array(self.data[company_name][self.data_size - self.window_size:], dtype=np.float32), axis=0)
            company_preds = []
            while curr_num_preds<num_predictions:
                y_preds = self.models[company_name].predict(input_data)
                curr_num_preds += len(y_preds)
                company_preds  = np.array(flat_append(company_preds, y_preds), dtype=np.float32)
                input_data = np.array(flat_append(np.squeeze(input_data), y_preds), dtype=np.float32)
                input_data = np.expand_dims(input_data[len(input_data)-self.window_size: ], axis=0)
            company_preds = np.array(company_preds, dtype=np.float32)[:num_predictions]
            predictions[company_name] = (X_future.flatten(), company_preds.copy())
        return predictions
                
    def train_NODE(self, hidden_dim=10, num_epochs=2000, lr=0.01):
        self.models = {}
        self.loss_fn = nn.MSELoss()
        self.optimizers = {}
        
        for company_name in self.data.keys():
            t = torch.linspace(0, 1, steps=len(self.data[company_name]))
            x = torch.tensor(self.data[company_name], dtype=torch.float32).unsqueeze(1)
            
            model = NODETimeSeries(input_dim=1, hidden_dim=hidden_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                x_pred = model(x[0], t)
                loss = self.loss_fn(x_pred.squeeze(), x.squeeze())
                loss.backward()
                optimizer.step()
                
                if epoch % 500 == 0:
                    print(f'Company: {company_name}, Epoch {epoch}, Loss: {loss.item()}')
            
            self.models[company_name] = model
            self.optimizers[company_name] = optimizer
    
    def predict_NODE(self, num_predictions=100):
        predictions = {}
        t_future = torch.linspace(1, 1 + num_predictions / len(self.data[list(self.data.keys())[0]]), steps=num_predictions)
        
        for company_name, model in self.models.items():
            with torch.no_grad():
                x_future = model(torch.tensor(self.data[company_name][-1], dtype=torch.float32).unsqueeze(0), t_future)
            predictions[company_name] = (np.arange(self.data_size + 1, self.data_size + num_predictions + 1), x_future.squeeze().numpy())
        
        return predictions
    
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

class Solver():
    def __init__(self, data, predicted_vals, risks):
        self.data = data
        self.predicted_vals = predicted_vals
        self.risks = risks

        self.current_prices = np.array([x[-1] for x in [self.data[y] for y in ORDER]])
        self.predicted_prices = np.array([x[-1] for x in [self.predicted_vals[y][1] for y in ORDER]])
        self.expected_returns = self.predicted_prices/self.current_prices - 1

        self.deltaf1 = 1
        self.deltaf2 = 1

        # Calculate the normalization parameters
        sol1 = self._solve_wsm(1.0, 0.0)
        sol2 = self._solve_wsm(0.0, 1.0)
        f11, f12 = self.get_objective_values(np.array(sol1["x"]).reshape(-1))
        f21, f22 = self.get_objective_values(np.array(sol2["x"]).reshape(-1))
        self.minf1 = min(f11, f21)
        self.minf2 = min(f12, f22)
        self.maxf1 = max(f11, f21)
        self.maxf2 = max(f12, f22)

        self.deltaf1 = self.maxf1 - self.minf1
        self.deltaf2 = self.maxf2 - self.minf2

    def _solve_wsm(self, w1, w2):
        n = len(self.expected_returns)

        # Transpose because of library and *2 because of the formula and normalize TODO verify
        Q = 2 * w1 * matrix(self.risks.T) / self.deltaf1
        c = -w2 * matrix(self.expected_returns) / self.deltaf2
        # Q = 2 * w1 * matrix(self.risks.T)
        # c = -w2 * matrix(self.expected_returns)

        # print(self.deltaf1, self.deltaf2)

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

        solutions = []
        for w1, w2 in zip(weights1, weights2):
            print(f"Running for w1={w1}, w2={w2}")
            sol = self._solve_wsm(float(w1), float(w2))
            sol_weights = np.array(sol["x"])
            sol_weights = sol_weights.reshape(-1)
            
            f1, f2 = self.get_objective_values(sol_weights)

            solutions.append((f1, f2, sol_weights))
        # print(sum(sol_weights), f1, f2)

        return solutions    
    
    def solve_ecm(self, num_thresholds=11):
        solutions = []

        thresholds = self.minf2 + np.linspace(0.0, 1.0, num_thresholds) * (self.deltaf2)

        for threshold in thresholds:
            print(f"Running for threshold={threshold}")
            sol = self._solve_ecm(threshold)
            sol_weights = np.array(sol["x"])
            sol_weights = sol_weights.reshape(-1)
            
            f1, f2 = self.get_objective_values(sol_weights)

            solutions.append((f1, f2, sol_weights))

        return solutions

    def _solve_ecm(self, return_threshold):
        n = len(self.expected_returns)

        # Transpose because of library and *2 because of the formula and normalize TODO verify
        Q = 2 * matrix(self.risks.T)
        c = matrix(np.zeros(n))

        # A = matrix(np.ones((1, n)), (1, n), 'd')
        # b = matrix(1.0)
        
        A = matrix(np.ones((1, n)))
        b = matrix(1.0)

        G = matrix(np.vstack((-np.array(self.expected_returns), -np.eye(n), np.eye(n))))
        h = matrix(np.hstack((-return_threshold, np.zeros(n), np.ones(n))))  

        sol = solvers.qp(Q, c, G, h, A, b)

        return sol
    
    def get_objective_values(self, solution_weights):
        solution_weights = solution_weights / np.sum(solution_weights)
        f1 = sum(solution_weights * self.expected_returns)
        f2 = solution_weights @ self.risks @ solution_weights.T

        return f1, f2


    def _generate_uniform_weights(self, n=20, step=0.2):
        results = []

        def backtrack(index, remaining, current):
            """Backtracking to generate valid weight combinations."""
            if index == n - 1:
                current.append(remaining)
                if np.all(np.array(current) >= 0):
                    results.append(tuple(current))
                current.pop()
                return
            for w in np.arange(0, min(remaining, 1) + step, step):
                current.append(w)
                backtrack(index + 1, round(remaining - w, 10), current)
                current.pop()

        backtrack(0, 1.0, [])
        return results
    

    def plot_front(self, solutions, n=20, step=0.2):
        random_points = []        
        
        results = self._generate_uniform_weights(n, step)
        for res in results:  
            sol_weights = np.array(res)
            f1, f2 = self.get_objective_values(sol_weights)

            random_points.append((f1, f2))


        plt.figure(figsize=(10, 9))
        plt.scatter([x[0] for x in random_points], [x[1] for x in random_points], alpha=0.8)
        plt.scatter([x[0] for x in solutions], [x[1] for x in solutions], color="red", linewidths=3)

        plt.xlabel("Expected Return")
        plt.ylabel("Risk")
        plt.title("Sampled Decision Variables")
        plt.grid(True)
        plt.show()

    def plot_sampled_decision_variables(self, n=20, step=0.2):
        random_points = []        
        results = self._generate_uniform_weights(n, step)
        for res in results:  
            sol_weights = np.array(res)
            f1, f2 = self.get_objective_values(sol_weights)

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
