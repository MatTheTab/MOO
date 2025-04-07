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
import seaborn as sns
from pymoo.indicators.hv import HV
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import plotly.graph_objects as go
from scipy.stats import qmc

ORDER = ["SuperFuture", "Apples", "WorldNow", "Electronics123", "Photons", "SpaceNow", "PearPear",
         "PositiveCorrelation", "BetterTechnology", "ABCDE", "EnviroLike", "Moneymakers", "Fuel4",
         "MarsProject", "CPU-XYZ", "RoboticsX", "Lasers", "WaterForce", "SafeAndCare", "BetterTomorrow"]

device = "cuda" if torch.cuda.is_available() else "cpu"

def pretty_display(solutions):
    data = []
    
    for sol in solutions:
        expected_return, risk, weights = sol
        row = {"Expected Gain": expected_return, "Risk": risk}
        row.update({company: weight for company, weight in zip(ORDER, weights)})
        row["Sanity Check"] = sum(weights)
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def save_txt(solution, filename):
    with open(filename, 'w') as f:
        expected_return, risk, weights = solution
        line = f"{expected_return:.2f} {risk:.2f} " + " ".join(f"{w:.2f}" for w in weights)
        f.write(line)

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
                
    def train_NODE(self, hidden_dim=10, num_epochs=300, lr=0.01):
        print(f"Using device: {device}")
        self.models = {}
        self.loss_fn = nn.MSELoss()
        self.optimizers = {}
        
        for company_name in self.data.keys():
            t = torch.linspace(0, 1, steps=len(self.data[company_name]))
            x = torch.tensor(self.data[company_name], dtype=torch.float32).unsqueeze(1)
            
            model = NODETimeSeries(input_dim=1, hidden_dim=hidden_dim)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                x_pred = model(x[0].to(device), t.to(device))
                loss = self.loss_fn(x_pred.squeeze(), x.to(device).squeeze())
                loss.backward()
                optimizer.step()
                
                if epoch % 100 == 0:
                    print(f'Company: {company_name}, Epoch {epoch}, Loss: {loss.item()}')
            
            self.models[company_name] = model.to("cpu")
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

    def plot_multiple_predictions(self, predictions_list, titles, val_range=None, figsize=(15, 5)):
        if len(predictions_list) != len(titles):
            raise ValueError("The number of predictions lists must match the number of titles.")


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

            for j, (predictions, title) in enumerate(zip(predictions_list, titles)):
                if j == 0:
                    axes[i].plot(predictions[company_name][0], predictions[company_name][1], linestyle='-', markersize=3, label=title, color="red")
                else:
                    axes[i].plot(predictions[company_name][0], predictions[company_name][1], linestyle='--', markersize=3, label=title)

                
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

        thresholds = self.minf1 + np.linspace(0.0, 1.0, num_thresholds) * (self.deltaf1)

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

    def plot_multiple_fronts(self, solutions_list, titles, n=20, step=0.2):
        if len(solutions_list) != 6 or len(titles) != 6:
            raise ValueError("You must provide exactly 6 solutions lists and 6 titles.")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (solutions, ax) in enumerate(zip(solutions_list, axes)):
            random_points = []        
            results = self._generate_uniform_weights(n, step)
            
            for res in results:  
                sol_weights = np.array(res)
                f1, f2 = self.get_objective_values(sol_weights)
                random_points.append((f1, f2))
            
            ax.scatter([x[0] for x in random_points], [x[1] for x in random_points], alpha=0.8)
            
            if i < 3:
                ax.scatter([x[0] for x in solutions], [x[1] for x in solutions], color="blue", linewidths=3)
            else:
                ax.scatter([x[0] for x in solutions], [x[1] for x in solutions], color="red", linewidths=3)
            
            ax.set_xlabel("Expected Return")
            ax.set_ylabel("Risk")
            ax.set_title(titles[i])
            ax.grid(True)
        
        plt.tight_layout()
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

class ExperimentRunner:
    def __init__(self, solver):
        self.solver = solver
        self.results_WSM = None
        self.results_ECM = None

    def run_WSM(self, steps):
        all_results = []
        for step in steps:
            solutions = self.solver.solve_wsm(step=step)
            solutions_df = pretty_display(solutions)
            all_results.append(solutions_df)
        self.results_WSM = pd.concat(all_results, ignore_index=True)

    def run_ECM(self, thresholds):
        all_results = []
        for num_threshold in thresholds:
            solutions = self.solver.solve_ecm(num_thresholds=num_threshold)
            solutions_df = pretty_display(solutions)
            all_results.append(solutions_df)
        self.results_ECM = pd.concat(all_results, ignore_index=True)

    def summarize(self):
        def get_top_solutions(df, method_name):
            if df is None:
               return pd.DataFrame(columns=["Method", "Solution Signature", "Frequency"])
        
            df_no_risk_gain = df.drop(columns=["Expected Gain", "Risk", "Sanity Check"])
            df["Solution Signature"] = df_no_risk_gain.apply(tuple, axis=1)
            top_solutions = df["Solution Signature"].value_counts().head(10).reset_index()
            top_solutions.columns = ["Solution Signature", "Frequency"]
            top_solutions["Method"] = method_name
            return top_solutions
        
        top_wsm = get_top_solutions(self.results_WSM, "WSM")
        top_ecm = get_top_solutions(self.results_ECM, "ECM")
        return top_ecm, top_wsm

    def plot_results(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        def plot_histogram(ax, df, title):
            if df is None:
                ax.set_title(f"{title} - No Data")
                return
            
            solution_counts = df.drop(columns=["Expected Gain", "Risk", "Sanity Check"]).apply(tuple, axis=1).value_counts()
            sns.histplot(solution_counts, bins=30, kde=True, ax=ax)
            ax.set_title(f"{title} - Solution Occurrences")
            ax.set_xlabel("Occurrences")
            ax.set_ylabel("Frequency")

        plot_histogram(axes[0], self.results_WSM, "WSM")
        plot_histogram(axes[1], self.results_ECM, "ECM")

        plt.tight_layout()
        plt.show()

    def plot_risk_return_distribution(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        def plot_scatter_and_distributions(df, title, row_idx):
            if df is None:
                for col in range(3):
                    axes[row_idx, col].set_title(f"{title} - No Data")
                return
            
            sns.scatterplot(x=df["Expected Gain"], y=df["Risk"], ax=axes[row_idx, 0])
            axes[row_idx, 0].set_title(f"{title} - Risk vs Return")
            axes[row_idx, 0].set_xlabel("Expected Gain")
            axes[row_idx, 0].set_ylabel("Risk")

            sns.histplot(df["Risk"], kde=True, bins=30, ax=axes[row_idx, 1])
            axes[row_idx, 1].set_title(f"{title} - Risk Distribution")
            axes[row_idx, 1].set_xlabel("Risk")

            sns.histplot(df["Expected Gain"], kde=True, bins=30, ax=axes[row_idx, 2])
            axes[row_idx, 2].set_title(f"{title} - Expected Gain Distribution")
            axes[row_idx, 2].set_xlabel("Expected Gain")

        plot_scatter_and_distributions(self.results_WSM, "WSM", 0)
        plot_scatter_and_distributions(self.results_ECM, "ECM", 1)

        plt.tight_layout()
        plt.show()

    def plot_top_corporations(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        def plot_bar_chart(ax, df, title):
            if df is None or df.empty:
                ax.set_title(f"{title} - No Data")
                return

            # Drop non-corporation columns (ensure the remaining ones are numeric)
            corp_columns = [col for col in df.columns if col not in ["Expected Gain", "Risk", "Sanity Check"]]
            corp_df = df[corp_columns].apply(pd.to_numeric, errors="coerce")  # Ensure numeric values
            
            total_investment = corp_df.sum(axis=0).sort_values(ascending=False)

            sns.barplot(x=total_investment.values, y=total_investment.index, ax=ax)
            ax.set_title(f"{title} - Total Investment per Corporation")
            ax.set_xlabel("Total Investment")
            ax.set_ylabel("Corporation")

        plot_bar_chart(axes[0], self.results_WSM, "WSM")
        plot_bar_chart(axes[1], self.results_ECM, "ECM")

        plt.tight_layout()
        plt.show()


def bimodal_sample(size=1):
    means = [0.2, 0.8]
    std_devs = [0.1, 0.1]
    choices = np.random.choice([0, 1], size=size)
    samples = np.random.normal(loc=np.array(means)[choices], scale=np.array(std_devs)[choices])
    return np.clip(samples, 0, 1)

def transform_solution(solution):
    new_sol = []
    for example in solution:
        temp = []
        temp.append(example[0])
        temp.append(example[1])
        for val in example[-1]:
            temp.append(val)
        new_sol.append(temp.copy())
    return np.array(new_sol, dtype=np.float32)

def calculate_3d_hypervolume(population, ref_point=np.array([1.1, 1.1])):
    hv_value = HV(ref_point = ref_point)(population)
    return hv_value

def find_non_dominated(F):
    M = np.zeros((F.shape[0], F.shape[0]))
    for i in range(F.shape[0]):
        for j in range(F.shape[0]):
            if np.all(F[i] <= F[j]) and not np.all(F[i] == F[j]):
                M[i, j] = 1
            elif np.all(F[j] <= F[i]) and not np.all(F[j] == F[i]):
                M[i, j] = -1
    I = np.where(np.all(M >= 0, axis=1))[0]
    return I

def calculate_2d_hyperarea(population, ref_point=np.array([1.1, 1.1])):
    hv_value = HV(ref_point = ref_point)(population)
    return hv_value

def find_non_dominated(F, _F=None):
    M = np.zeros((F.shape[0],F.shape[0]))
    for i in range(F.shape[0]):
      for j in range(F.shape[0]):
        if np.all(F[i]<=F[j]) and not np.all(F[i]==F[j]):
          M[i,j] = 1
        elif np.all(F[j]<=F[i]) and not np.all(F[j]==F[i]):
          M[i,j] = -1
    I = np.where(np.all(M >= 0, axis=1))[0]
    return I

class GeneticSolver(Solver):
    def __init__(self, data, predicted_vals, risks, population_size, mutation_probability, weight_size=20, num_dimensions=2):
        super().__init__(data, predicted_vals, risks)
        self.population_size = population_size  
        self.weight_size = weight_size
        self.log_history = False
        self.num_dimensions = num_dimensions
        self.mutation_prob = mutation_probability

    def enable_loging(self):
        self.log_history = True
    
    def disable_loging(self):
        self.log_history = False

    def normalize(self, population):
        return population / population.sum(axis=1, keepdims=True)
    
    def normalize_population_values(self, population, num_dims=2):
        values = []
        for weights in population:
            if num_dims == 3:
                potential_gain = self.maxf1 - (weights.T @ self.expected_returns)
                risk = weights.T @ self.risks @ weights
                potential_gain = (potential_gain - self.minf1)/(self.maxf1 - self.minf1)
                num_zero_weights = np.sum(weights < 0.01)/self.weight_size
                risk = (risk - self.minf2)/(self.maxf2 - self.minf2)
                temp = [potential_gain, risk, num_zero_weights]
            else:
                potential_gain = self.maxf1 - (weights.T @ self.expected_returns)
                risk = weights.T @ self.risks @ weights
                potential_gain = (potential_gain - self.minf1)/(self.maxf1 - self.minf1)
                risk = (risk - self.minf2)/(self.maxf2 - self.minf2)
                temp = [potential_gain, risk]
            values.append(temp.copy())
        return np.array(values, dtype=np.float32)

    def generate_population(self):
        alpha = np.ones(self.weight_size)
        population = np.array([np.random.dirichlet(alpha) for _ in range(self.population_size)], dtype=np.float32)
        if self.log_history:
            self.population_history = []
            self.population_history.append(population.copy())
            
        return population
    
    def initialize_chebyshev(self):
        if self.num_dimensions <= 1:
            print("At least two dimensions must be present")
            raise ValueError
        elif self.num_dimensions == 2:
            samples_dim_1 = np.linspace(0, 1, self.population_size)
            samples_dim_2 = 1 - samples_dim_1
            weights = []
            for i in range(len(samples_dim_1)):
                weights.append([samples_dim_1[i], samples_dim_2[i]])
            weights = np.array(weights, dtype=np.float32)
        else:
            weights = np.random.dirichlet(np.ones(self.num_dimensions), self.population_size).astype(np.float32)
        return weights
    

    def crossover(self, chebyshev_weights, neighborhoods, weight_to_individual_map):
        np.random.shuffle(chebyshev_weights)
        weight = chebyshev_weights[0]
        neighboring_weights = neighborhoods[tuple(weight)].copy()
        random.shuffle(neighboring_weights)
        weight_parent_1, weight_parent_2 = neighboring_weights[:2]
        parent_1, parent_2 = weight_to_individual_map[weight_parent_1], weight_to_individual_map[weight_parent_2]
        alpha = np.random.rand()
        offspring = alpha * parent_1 + (1 - alpha) * parent_2
        offspring = offspring / np.sum(offspring)
        
        return offspring, weight
    
    def mutate(self, individual):
        if random.random() < self.mutation_prob:
            idx = random.randint(0, len(individual) - 1)
            individual[idx] = random.random()
        return individual/np.sum(individual)
    
    def find_neighborhoods(self, chebyshev_weights, neighborhood_size):
        distances = np.abs(chebyshev_weights[:, None] - chebyshev_weights).sum(axis=-1)
        neighbors_dict = {}
        for i, w in enumerate(chebyshev_weights):
            sorted_indices = np.argsort(distances[i])
            closest_indices = sorted_indices[1:neighborhood_size+1]
            neighbors_dict[tuple(w)] = [tuple(chebyshev_weights[j]) for j in closest_indices]
        return neighbors_dict
    
    def assign_chebyshev_weights(self, population, chebyshev_weights):
        np.random.shuffle(chebyshev_weights)
        weight_to_individual_map = {tuple(chebyshev_weights[i]): population[i] for i in range(self.population_size)}
        return weight_to_individual_map
    
    def calculate_chebyshev_distance(self, weight, investment_vector):
        potential_gain = self.maxf1 - (investment_vector.T @ self.expected_returns)
        #risk = np.sqrt(investment_vector.T @ self.risks @ investment_vector)
        risk = investment_vector.T @ self.risks @ investment_vector
        potential_gain = (potential_gain - self.minf1)/(self.maxf1 - self.minf1)
        risk = (risk - self.minf2)/(self.maxf2 - self.minf2)
        return max(weight[0]*potential_gain, weight[1]*risk)
    
    def calculate_chebyshev_3d_distance(self, weight, investment_vector):
        potential_gain = self.maxf1 - (investment_vector.T @ self.expected_returns)
        #risk = np.sqrt(investment_vector.T @ self.risks @ investment_vector)
        risk = investment_vector.T @ self.risks @ investment_vector
        potential_gain = (potential_gain - self.minf1)/(self.maxf1 - self.minf1)
        num_zero_weights = np.sum(investment_vector < 0.01)/self.weight_size
        risk = (risk - self.minf2)/(self.maxf2 - self.minf2)
        return max(weight[0]*potential_gain, weight[1]*risk, weight[2] * num_zero_weights)
    
    def perform_selection(self, offspring, weight_to_individual_map, origin_weight, neighborhoods):
        #Spaghetti code, but I do not care as long as it works
        complete_pool = []
        for weight in neighborhoods[tuple(origin_weight)]:
            complete_pool.append(weight_to_individual_map[tuple(weight)])
        complete_pool.append(offspring)
        complete_pool = np.array(complete_pool, dtype=np.float32)
        next_generation = []
        for weight in neighborhoods[tuple(origin_weight)]:
            current_best = weight_to_individual_map[tuple(weight)]
            if self.num_dimensions == 3:
                chebyshev_distance_current = self.calculate_chebyshev_3d_distance(weight, weight_to_individual_map[tuple(weight)])
            else:
                chebyshev_distance_current = self.calculate_chebyshev_distance(weight, weight_to_individual_map[tuple(weight)])
            if self.num_dimensions == 3:
                chebyshev_distance_new = self.calculate_chebyshev_3d_distance(weight, offspring)
            else:
                chebyshev_distance_new = self.calculate_chebyshev_distance(weight, offspring)
            if chebyshev_distance_new < chebyshev_distance_current:
                current_best = offspring
                chebyshev_distance_current = chebyshev_distance_new
            weight_to_individual_map[tuple(weight)] = current_best
        for key in weight_to_individual_map:
            next_generation.append(weight_to_individual_map[key])
        return np.array(next_generation, dtype=np.float32), weight_to_individual_map

    def genetic_optimize(self, generations=100, neighborhood_size=2):
        generations = generations*self.population_size
        chebyshev_weights = self.initialize_chebyshev()
        neighborhoods = self.find_neighborhoods(chebyshev_weights, neighborhood_size)
        population = self.generate_population()
        weight_to_individual_map = self.assign_chebyshev_weights(population, chebyshev_weights)
        for generation in range(generations):
            offspring, origin_weight = self.crossover(chebyshev_weights, neighborhoods, weight_to_individual_map)
            offspring = self.mutate(offspring)
            population, weight_to_individual_map = self.perform_selection(offspring, weight_to_individual_map, origin_weight, neighborhoods)
            if self.log_history:
                self.population_history.append(self.normalize(population.copy()))
        return population
    
    def get_solutions_from_population(self, population):
        solutions = []
        for individual in population:
            solution = []
            potential_gain = individual.T @ self.expected_returns
            #risk = np.sqrt(individual.T @ self.risks @ individual)
            risk = individual.T @ self.risks @ individual
            solution.append(potential_gain)
            solution.append(risk)
            solution.append(individual.copy())
            solutions.append(solution.copy())
        return solutions
    
    def get_solutions_from_population_3D(self, population):
        solutions = []
        for individual in population:
            solution = []
            potential_gain = individual.T @ self.expected_returns
            #risk = np.sqrt(individual.T @ self.risks @ individual)
            risk = individual.T @ self.risks @ individual
            num_zero_weights = np.sum(individual < 0.01)/self.weight_size
            solution.append(potential_gain)
            solution.append(risk)
            solution.append(num_zero_weights)
            solution.append(individual.copy())
            solutions.append(solution.copy())
        return solutions

    def plot_sampled_3D(self, n=20, step=0.2, color=True):
        random_points = []        
        results = self._generate_uniform_weights(n, step)
        
        for res in results:  
            sol_weights = np.array(res)
            f1, f2 = self.get_objective_values(sol_weights)  # Assuming f3 exists
            f3 = np.sum(sol_weights < 0.01)/self.weight_size
            random_points.append((f1, f2, f3))
        
        if color:
            plt.figure(figsize=(10, 9))
            scatter = plt.scatter(
                [x[0] for x in random_points], 
                [x[1] for x in random_points], 
                c=[x[2] for x in random_points],  # Color by f3
                cmap='viridis', 
                edgecolors='k'
            )
            plt.colorbar(scatter, label='Non-Zero Weights')
            plt.xlabel("Expected Return")
            plt.ylabel("Risk")
            plt.title("Sampled Decision Variables")
            plt.grid(True)
            plt.show()
        else:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            axs[0].scatter([x[0] for x in random_points], [x[1] for x in random_points])
            axs[0].set_xlabel("Expected Gain")
            axs[0].set_ylabel("Risk")
            axs[0].set_title("Expected Gain vs Risk")
            
            axs[1].scatter([x[1] for x in random_points], [x[2] for x in random_points])
            axs[1].set_xlabel("Risk")
            axs[1].set_ylabel("Non-Zero Weights")
            axs[1].set_title("Risk vs Non-Zero Weights")
            
            axs[2].scatter([x[0] for x in random_points], [x[2] for x in random_points])
            axs[2].set_xlabel("Expected Gain")
            axs[2].set_ylabel("Non-Zero Weights")
            axs[2].set_title("Expected Gain vs Non-Zero Weights")
            
            for ax in axs:
                ax.grid(True)
            
            plt.tight_layout()
            plt.show()

    def plot_front_3D(self, solutions, n=20, step=0.2, color=True):
        random_points = []        
        results = self._generate_uniform_weights(n, step)
        
        for res in results:  
            sol_weights = np.array(res)
            f1, f2 = self.get_objective_values(sol_weights)  # Assuming f3 exists
            f3 = np.sum(sol_weights < 0.01)/self.weight_size
            random_points.append((f1, f2, f3))
        
        if color:
            plt.figure(figsize=(10, 9))
            scatter = plt.scatter(
                [x[0] for x in random_points], 
                [x[1] for x in random_points], 
                c=[x[2] for x in random_points],  # Color by f3
                cmap='viridis', 
                edgecolors='k',
                alpha=0.3
            )
            plt.colorbar(scatter, label='Non-Zero Weights')
            
            solution_colors = [x[2] for x in solutions]  # Color solutions by their f3 values
            sol_scatter = plt.scatter(
                [x[0] for x in solutions], 
                [x[1] for x in solutions], 
                c=solution_colors, 
                cmap='coolwarm',  # Different gradient
                edgecolors='black',
                linewidths=1.5,
                label='Front Solutions'
            )
            plt.colorbar(sol_scatter, label='Solution Non-Zero Weights')
            
            plt.xlabel("Expected Return")
            plt.ylabel("Risk")
            plt.title("Sampled Decision Variables")
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            axs[0].scatter([x[0] for x in random_points], [x[1] for x in random_points], alpha=0.8)
            axs[0].scatter([x[0] for x in solutions], [x[1] for x in solutions], color="red", linewidths=2)
            axs[0].set_xlabel("Expected Gain")
            axs[0].set_ylabel("Risk")
            axs[0].set_title("Expected Gain vs Risk")
            
            axs[1].scatter([x[1] for x in random_points], [x[2] for x in random_points], alpha=0.8)
            axs[1].scatter([x[1] for x in solutions], [x[2] for x in solutions], color="red", linewidths=2)
            axs[1].set_xlabel("Risk")
            axs[1].set_ylabel("Non-Zero Weights")
            axs[1].set_title("Risk vs Non-Zero Weights")
            
            axs[2].scatter([x[0] for x in random_points], [x[2] for x in random_points], alpha=0.8)
            axs[2].scatter([x[0] for x in solutions], [x[2] for x in solutions], color="red", linewidths=2)
            axs[2].set_xlabel("Expected Gain")
            axs[2].set_ylabel("Non-Zero Weights")
            axs[2].set_title("Expected Gain vs Non-Zero Weights")
            
            for ax in axs:
                ax.grid(True)
            
            plt.tight_layout()
            plt.show()

    def plot_many_3D(self, solutions1, solutions2, solutions3, n=20, step=0.2):
        random_points = []
        
        # Generate uniform weights for the front
        results = self._generate_uniform_weights(n, step)
        for res in results:
            sol_weights = np.array(res)
            f1, f2 = self.get_objective_values(sol_weights)  # Assuming f3 exists
            f3 = np.sum(sol_weights < 0.01) / self.weight_size
            random_points.append((f1, f2, f3))
        
        # Create subplots (2D)
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))

        # First subplot: Solutions 1 with 'viridis' color map (representing f3)
        scatter1 = axs[0].scatter(
            [x[0] for x in random_points],
            [x[1] for x in random_points],
            c=[x[2] for x in random_points],  # Color by f3
            cmap='viridis',
            alpha=0.6
        )
        axs[0].scatter(
            [x[0] for x in solutions1],
            [x[1] for x in solutions1],
            c=[x[2] for x in solutions1],
            cmap='viridis',  # Same color map for solutions
            edgecolors='k',
            linewidths=2
        )
        axs[0].set_xlabel("Expected Return")
        axs[0].set_ylabel("Risk")
        axs[0].set_title("Solution Population 1 (Viridis)")
        fig.colorbar(scatter1, ax=axs[0], label='Non-Zero Weights')

        # Second subplot: Solutions 2 with 'plasma' color map (representing f3)
        scatter2 = axs[1].scatter(
            [x[0] for x in random_points],
            [x[1] for x in random_points],
            c=[x[2] for x in random_points],  # Color by f3
            cmap='plasma',
            alpha=0.6
        )
        axs[1].scatter(
            [x[0] for x in solutions2],
            [x[1] for x in solutions2],
            c=[x[2] for x in solutions2],
            cmap='plasma',  # Same color map for solutions
            edgecolors='k',
            linewidths=2
        )
        axs[1].set_xlabel("Expected Return")
        axs[1].set_ylabel("Risk")
        axs[1].set_title("Solution Population 2 (Plasma)")
        fig.colorbar(scatter2, ax=axs[1], label='Non-Zero Weights')

        # Third subplot: Solutions 3 with 'inferno' color map (representing f3)
        scatter3 = axs[2].scatter(
            [x[0] for x in random_points],
            [x[1] for x in random_points],
            c=[x[2] for x in random_points],  # Color by f3
            cmap='cool',
            alpha=0.6
        )
        axs[2].scatter(
            [x[0] for x in solutions3],
            [x[1] for x in solutions3],
            c=[x[2] for x in solutions3],
            cmap='cool',  # Same color map for solutions
            edgecolors='k',
            linewidths=2
        )
        axs[2].set_xlabel("Expected Return")
        axs[2].set_ylabel("Risk")
        axs[2].set_title("Solution Population 3 (Cool)")
        fig.colorbar(scatter3, ax=axs[2], label='Non-Zero Weights')

        plt.tight_layout()
        plt.show()

    def plot_front_real_3D(self, solutions, color=True, colorscale="agsunset"):
        fig = go.Figure()

        if color:
            solution_colors = [x[2] for x in solutions]
            fig.add_trace(go.Scatter3d(
                x=[x[0] for x in solutions], 
                y=[x[1] for x in solutions], 
                z=[x[2] for x in solutions], 
                mode='markers',
                marker=dict(
                    size=6,
                    color=solution_colors,
                    colorscale=colorscale,
                    cmin=min(solution_colors),
                    cmax=max(solution_colors),
                    colorbar=dict(title="Solution Non-Zero Weights"),
                    line=dict(width=1, color='black')
                ),
                name='Front Solutions'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[x[0] for x in solutions], 
                y=[x[1] for x in solutions], 
                z=[x[2] for x in solutions], 
                mode='markers',
                marker=dict(
                    size=6,
                    color='red',
                    line=dict(width=1, color='black')
                ),
                name='Front Solutions'
            ))

        # Update layout for better visualization
        fig.update_layout(
            title="3D Front Visualization",
            scene=dict(
                xaxis_title="Expected Return",
                yaxis_title="Risk",
                zaxis_title="Non-Zero Weights"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        fig.show()


    def plot_fronts_heatmap(self, populations):
        plt.figure(figsize=(10, 9))
        cmap = cm.get_cmap("coolwarm", len(populations))  # Color gradient
        
        for gen_idx, solutions in enumerate(populations):
            color = cmap(gen_idx / len(populations))  # Normalize color scale
            
            # Extracting front points
            front_x = [x[0] for x in solutions]
            front_y = [x[1] for x in solutions]
            
            # Plot the front with varying brightness
            plt.scatter(front_x, front_y, color=color, label=f'Gen {gen_idx + 1}', alpha=0.8)
        
        plt.xlabel("Expected Return")
        plt.ylabel("Risk")
        plt.title("Pareto Front Evolution Over Generations")
        plt.grid(True)
        plt.show()

    def plot_many(self, solutions1, solutions2, solutions3, n=20, step=0.2):
        random_points = []
        
        # Generate uniform weights for the front
        results = self._generate_uniform_weights(n, step)
        for res in results:
            sol_weights = np.array(res)
            f1, f2 = self.get_objective_values(sol_weights)
            random_points.append((f1, f2))
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # First subplot: Solutions 1 with red color
        axes[0].scatter([x[0] for x in random_points], [x[1] for x in random_points], alpha=0.8)
        axes[0].scatter([x[0] for x in solutions1], [x[1] for x in solutions1], color="red", linewidths=3)
        axes[0].set_xlabel("Expected Return")
        axes[0].set_ylabel("Risk")
        axes[0].set_title("Solution Population 1")
        axes[0].grid(True)
        
        # Second subplot: Solutions 2 with green color
        axes[1].scatter([x[0] for x in random_points], [x[1] for x in random_points], alpha=0.8)
        axes[1].scatter([x[0] for x in solutions2], [x[1] for x in solutions2], color="green", linewidths=3)
        axes[1].set_xlabel("Expected Return")
        axes[1].set_ylabel("Risk")
        axes[1].set_title("Solution Population 2")
        axes[1].grid(True)
        
        # Third subplot: Solutions 3 with orange color
        axes[2].scatter([x[0] for x in random_points], [x[1] for x in random_points], alpha=0.8)
        axes[2].scatter([x[0] for x in solutions3], [x[1] for x in solutions3], color="orange", linewidths=3)
        axes[2].set_xlabel("Expected Return")
        axes[2].set_ylabel("Risk")
        axes[2].set_title("Solution Population 3")
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

class GeneticSolverGuided(GeneticSolver):
    def __init__(self, data, predicted_vals, risks, population_size, mutation_probability, weight_size=20, num_dimensions=2):
        super().__init__(data, predicted_vals, risks, population_size, mutation_probability, weight_size, num_dimensions)

    def crossover(self, chebyshev_weights, neighborhoods, weight_to_individual_map):
        np.random.shuffle(chebyshev_weights)
        weight = chebyshev_weights[0]
        neighboring_weights = neighborhoods[tuple(weight)].copy()
        random.shuffle(neighboring_weights)
        weight_parent_1, weight_parent_2 = neighboring_weights[:2]
        parent_1, parent_2 = weight_to_individual_map[weight_parent_1], weight_to_individual_map[weight_parent_2]
        alpha = np.random.rand()
        offspring = alpha * parent_1 + (1 - alpha) * parent_2
        offspring = offspring / np.sum(offspring)
        return offspring, weight
    
    def mutate(self, individual):
        if random.random() < self.mutation_prob:
            idx_1 = random.randint(0, len(individual) - 1)
            individual[idx_1] = 1.0
        return individual/np.sum(individual)
    
    def perform_selection(self, offspring, weight_to_individual_map, origin_weight, neighborhoods):
        complete_pool = []
        for weight in neighborhoods[tuple(origin_weight)]:
            complete_pool.append(weight_to_individual_map[tuple(weight)])
        complete_pool.append(offspring)
        complete_pool = np.array(complete_pool, dtype=np.float32)
        next_generation = []
        for weight in neighborhoods[tuple(origin_weight)]:
            current_best = weight_to_individual_map[tuple(weight)]
            if self.num_dimensions == 3:
                chebyshev_distance_current = self.calculate_chebyshev_3d_distance(weight, weight_to_individual_map[tuple(weight)])
            else:
                chebyshev_distance_current = self.calculate_chebyshev_distance(weight, weight_to_individual_map[tuple(weight)])
            if self.num_dimensions == 3:
                chebyshev_distance_new = self.calculate_chebyshev_3d_distance(weight, offspring)
            else:
                chebyshev_distance_new = self.calculate_chebyshev_distance(weight, offspring)
            if chebyshev_distance_new < chebyshev_distance_current:
                current_best = offspring
                chebyshev_distance_current = chebyshev_distance_new
            weight_to_individual_map[tuple(weight)] = current_best
        for key in weight_to_individual_map:
            next_generation.append(weight_to_individual_map[key])
        return np.array(next_generation, dtype=np.float32), weight_to_individual_map
    
    def initialize_deltas(self, weight_to_individual_map):
        deltas = {}
        for weight in weight_to_individual_map:
            deltas[weight] = []
        return deltas
    
    def update_deltas(self, deltas, weight_to_individual_map):
        for weight in weight_to_individual_map:
            deltas[weight].append(weight_to_individual_map[weight].copy())
        return deltas

    def calculate_deltas(self, delta_list):
        if not delta_list or len(delta_list) < 2:
            return np.zeros(len(delta_list[0]) if delta_list else 0)

        prev = None
        last = None

        # Iterate from the end to find the last two different vectors
        for vec in reversed(delta_list):
            vec = np.array(vec)
            if last is None:
                last = vec
            elif not np.array_equal(vec, last):
                prev = vec
                break

        if prev is None:
            return np.zeros(len(last))  # No meaningful change

        return last - prev

    def mutate_with_delta(self, parent, avg_delta, scale=0.1):
        parent = np.array(parent)
        avg_delta = np.array(avg_delta)
        noise = np.random.normal(loc=avg_delta, scale=scale, size=len(parent))
        mutated = parent + noise
        mutated = np.clip(mutated, 0, 1)
        total = np.sum(mutated)
        if total == 0:
            mutated = np.ones_like(mutated) / len(mutated)
        else:
            mutated /= total
        return mutated

    def deltas_mutate(self, offspring, deltas, origin_weight, neighborhoods, weight_to_individual_map):
        neighboring_weights = neighborhoods[tuple(origin_weight)]
        if random.random() < self.mutation_prob:
            # You can still use distance-based neighbor selection if you want:
            closest_neighbor = min(
                neighboring_weights,
                key=lambda neighbor: np.linalg.norm(offspring - weight_to_individual_map[neighbor])
            )

            # closest_neighbor = random.choice(neighboring_weights)

            delta = self.calculate_deltas(deltas[tuple(closest_neighbor)])
            offspring = self.mutate_with_delta(offspring, delta, scale=0.5)

        return offspring



    def genetic_optimize(self, generations=100, neighborhood_size=2, gen_threshold=5):
        generations = generations*self.population_size
        gen_threshold = gen_threshold * self.population_size
        chebyshev_weights = self.initialize_chebyshev()
        neighborhoods = self.find_neighborhoods(chebyshev_weights, neighborhood_size)
        population = self.generate_population()
        weight_to_individual_map = self.assign_chebyshev_weights(population, chebyshev_weights)
        deltas = self.initialize_deltas(weight_to_individual_map)

        for generation in range(generations):
            offspring, origin_weight = self.crossover(chebyshev_weights, neighborhoods, weight_to_individual_map)
            if generation < gen_threshold:
                offspring = self.mutate(offspring)
            else:
                if random.random() < 0.5:
                    offspring = self.deltas_mutate(offspring, deltas, origin_weight, neighborhoods, weight_to_individual_map)
                else:
                    offspring = self.mutate(offspring)
            population, weight_to_individual_map = self.perform_selection(offspring, weight_to_individual_map, origin_weight, neighborhoods)
            deltas = self.update_deltas(deltas, weight_to_individual_map)
            if self.log_history:
                self.population_history.append(self.normalize(population.copy()))
        return population


class GeneticExperimentRunner():
    def get_hypervolume_vs_population_size_vs_generations(self, pop_sizes, generations, data, predictions, risks):
        results = []
        for num_dims in [2, 3]:
            for pop_size in pop_sizes:
                for generation in generations:
                    GS = GeneticSolver(data, predictions, risks, population_size=100, mutation_probability=0.8, weight_size=20, num_dimensions=num_dims)
                    population = GS.genetic_optimize(generations=generation, neighborhood_size=5)
                    if num_dims == 2:
                        ref_point = np.array([1.1, 1.1], dtype=np.float32)
                    else:
                        ref_point = np.array([1.1, 1.1, 1.1], dtype=np.float32)
                    population_values = GS.normalize_population_values(population, num_dims=num_dims)
                    if num_dims == 2:
                        hv_value = calculate_2d_hyperarea(population_values, ref_point)
                    else:
                        hv_value = calculate_3d_hypervolume(population_values, ref_point)
                    results.append([num_dims, pop_size, generation, hv_value])
        self.results = results

    def plot_hypervolume_vs_population_size_vs_generations(self):
        results = np.array(self.results)
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        
        for num_dims, ax in zip([2, 3], axes):
            subset = results[results[:, 0] == num_dims]
            
            for generation in np.unique(subset[:, 2]):
                filtered = subset[subset[:, 2] == generation]
                ax.plot(
                    filtered[:, 1],
                    filtered[:, 3],
                    marker='o',
                    linestyle='-',
                    label=f"Generations: {int(generation)}"
                )
            
            ax.set_title(f"Hypervolume vs Population Size (Dimensions: {num_dims})")
            ax.set_xlabel("Population Size")
            ax.set_ylabel("Hypervolume" if num_dims == 3 else "Hyperarea")
            ax.legend()
        
        plt.tight_layout()
        plt.show()

    def get_genetic_vs_classic(self, data, predictions, risks, num_retries=10):
        solver = Solver(data, predictions, risks)
        GS = GeneticSolver(data, predictions, risks, population_size=None, mutation_probability=0.8, weight_size=20, num_dimensions=2)
        self.wsm_solutions_step_001 = transform_solution(solver.solve_wsm(step = 0.01))
        self.wsm_solutions_step_025 = transform_solution(solver.solve_wsm(step = 0.05))
        self.wsm_solutions_step_010 = transform_solution(solver.solve_wsm(step = 0.1))
        self.ecm_solutions_100 = transform_solution(solver.solve_ecm(num_thresholds=101))
        self.ecm_solutions_20 = transform_solution(solver.solve_ecm(num_thresholds=21))
        self.ecm_solutions_10 = transform_solution(solver.solve_ecm(num_thresholds=11))
        
        self.wsm_solutions_step_001_hv = calculate_2d_hyperarea(GS.normalize_population_values(self.wsm_solutions_step_001[:, 2: ], num_dims = 2))
        self.wsm_solutions_step_025_hv = calculate_2d_hyperarea(GS.normalize_population_values(self.wsm_solutions_step_025[:, 2: ], num_dims = 2))
        self.wsm_solutions_step_010_hv = calculate_2d_hyperarea(GS.normalize_population_values(self.wsm_solutions_step_010[:, 2: ], num_dims = 2))
        self.ecm_solutions_100_hv = calculate_2d_hyperarea(GS.normalize_population_values(self.ecm_solutions_100[:, 2: ], num_dims = 2))
        self.ecm_solutions_20_hv = calculate_2d_hyperarea(GS.normalize_population_values(self.ecm_solutions_20[:, 2: ], num_dims = 2))
        self.ecm_solutions_10_hv = calculate_2d_hyperarea(GS.normalize_population_values(self.ecm_solutions_10[:, 2: ], num_dims = 2))
        results = []
        front = []
        for retry in range(num_retries):
            for pop_size in [10, 20, 100]:
                GS1 = GeneticSolver(data, predictions, risks, population_size=pop_size, mutation_probability=0.8, weight_size=20, num_dimensions=2)
                GS1.enable_loging()
                GS1.genetic_optimize(generations=150, neighborhood_size=5)

                GS2 = GeneticSolver(data, predictions, risks, population_size=pop_size, mutation_probability=0.8, weight_size=20, num_dimensions=2)
                GS2.enable_loging()
                GS2.genetic_optimize(generations=150, neighborhood_size=5)

                GS3 = GeneticSolver(data, predictions, risks, population_size=pop_size, mutation_probability=0.8, weight_size=20, num_dimensions=2)
                GS3.enable_loging()
                GS3.genetic_optimize(generations=150, neighborhood_size=5)


                population_history_1 = GS1.population_history
                population_history_2 = GS2.population_history
                population_history_3 = GS3.population_history

                for pop_index, population in enumerate(population_history_1):
                    ref_point = np.array([1.1, 1.1], dtype=np.float32)
                    population_values = GS.normalize_population_values(population, num_dims=2)
                    hv_value = calculate_2d_hyperarea(population_values, ref_point)
                    results.append(["population_1", pop_size, pop_index, retry, hv_value])

                for pop_index, population in enumerate(population_history_1):
                    ref_point = np.array([1.1, 1.1], dtype=np.float32)
                    population_values = GS.normalize_population_values(population, num_dims=2)
                    hv_value = calculate_2d_hyperarea(population_values, ref_point)
                    results.append(["population_2", pop_size, pop_index, retry, hv_value])

                for pop_index, population in enumerate(population_history_1):
                    ref_point = np.array([1.1, 1.1], dtype=np.float32)
                    population_values = GS.normalize_population_values(population, num_dims=2)
                    hv_value = calculate_2d_hyperarea(population_values, ref_point)
                    results.append(["population_3", pop_size, pop_index, retry, hv_value])


        self.results = results

    def plot_genetic_vs_classic(self):
        results_df = pd.DataFrame(self.results, columns=['population', 'pop_size', 'generation', 'retry', 'hv_value'])
        wsm_hvs = [
            np.mean(self.wsm_solutions_step_001_hv),
            np.mean(self.wsm_solutions_step_025_hv),
            np.mean(self.wsm_solutions_step_010_hv)
        ]
        ecm_hvs = [
            np.mean(self.ecm_solutions_100_hv),
            np.mean(self.ecm_solutions_20_hv),
            np.mean(self.ecm_solutions_10_hv)
        ]
        
        pop_sizes = [10, 20, 100]
        population_colors = {'population_1': 'blue', 'population_2': 'green', 'population_3': 'orange'}
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sns.set_style("whitegrid")
        y_min = results_df['hv_value'].min()
        y_max = results_df['hv_value'].max()

        for i, pop_size in enumerate(pop_sizes):
            subset = results_df[results_df['pop_size'] == pop_size]
            subset['scaled_generation'] = subset['generation'] / pop_size
            stats = subset.groupby('scaled_generation')['hv_value'].agg(['mean', 'std'])

            for population, color in population_colors.items():
                pop_subset = subset[subset['population'] == population]
                pop_stats = pop_subset.groupby('scaled_generation')['hv_value'].agg(['mean', 'std'])
                axes[i].plot(pop_stats.index, pop_stats['mean'], linestyle='-', label=f'{population} Mean HV', color=color)
                axes[i].fill_between(pop_stats.index, pop_stats['mean'] - pop_stats['std'], pop_stats['mean'] + pop_stats['std'], 
                                    color=color, alpha=0.2, label=f'{population} ±1 Std Dev')
                
            axes[i].axhline(y=wsm_hvs[i], color='r', linestyle='--', label='WSM Hypervolume')
            axes[i].axhline(y=ecm_hvs[i], color='b', linestyle='--', label='ECM Hypervolume')
            axes[i].set_title(f'Convergence for Pop Size {pop_size}')
            axes[i].set_xlabel('Generation (scaled)')
            axes[i].set_ylabel('Hyperarea')
            axes[i].set_ylim(0.1, 1.2)
            axes[i].legend()
            
        plt.tight_layout()
        plt.show()


    def get_convergence(self, data, predictions, risks, num_retries=10):
        results = []
        for num_dims in [2, 3]:
            for retry in range(num_retries):
                GS1 = GeneticSolver(data, predictions, risks, population_size=100, mutation_probability=0.8, weight_size=20, num_dimensions=num_dims)
                GS1.enable_loging()
                GS1.genetic_optimize(generations=150, neighborhood_size=5)

                GS2 = GeneticSolver(data, predictions, risks, population_size=100, mutation_probability=0.8, weight_size=20, num_dimensions=num_dims)
                GS2.enable_loging()
                GS2.genetic_optimize(generations=150, neighborhood_size=5)

                GS3 = GeneticSolver(data, predictions, risks, population_size=100, mutation_probability=0.8, weight_size=20, num_dimensions=num_dims)
                GS3.enable_loging()
                GS3.genetic_optimize(generations=150, neighborhood_size=5)

                population_history_1 = GS1.population_history
                population_history_2 = GS2.population_history
                population_history_3 = GS3.population_history

                for pop_index, population in enumerate(population_history_1):
                    if num_dims == 2:
                        ref_point = np.array([1.1, 1.1], dtype=np.float32)
                    else:
                        ref_point = np.array([1.1, 1.1, 1.1], dtype=np.float32)
                    population_values = GS1.normalize_population_values(population, num_dims=num_dims)
                    if num_dims == 2:
                        hv_value = calculate_2d_hyperarea(population_values, ref_point)
                    else:
                        hv_value = calculate_3d_hypervolume(population_values, ref_point)
                    results.append(["population_1", num_dims, pop_index, retry, hv_value])



                for pop_index, population in enumerate(population_history_2):
                    if num_dims == 2:
                        ref_point = np.array([1.1, 1.1], dtype=np.float32)
                    else:
                        ref_point = np.array([1.1, 1.1, 1.1], dtype=np.float32)
                    population_values = GS1.normalize_population_values(population, num_dims=num_dims)
                    if num_dims == 2:
                        hv_value = calculate_2d_hyperarea(population_values, ref_point)
                    else:
                        hv_value = calculate_3d_hypervolume(population_values, ref_point)
                    results.append(["population_2", num_dims, pop_index, retry, hv_value])



                for pop_index, population in enumerate(population_history_3):
                    if num_dims == 2:
                        ref_point = np.array([1.1, 1.1], dtype=np.float32)
                    else:
                        ref_point = np.array([1.1, 1.1, 1.1], dtype=np.float32)
                    population_values = GS1.normalize_population_values(population, num_dims=num_dims)
                    if num_dims == 2:
                        hv_value = calculate_2d_hyperarea(population_values, ref_point)
                    else:
                        hv_value = calculate_3d_hypervolume(population_values, ref_point)
                    results.append(["population_3", num_dims, pop_index, retry, hv_value])

        self.results = results

    def plot_convergence(self):
        results = self.results
        results_df = pd.DataFrame(results, columns=['population', 'num_dims', 'generation', 'retry', 'hv_value'])

        # Set up the figure for convergence plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.set_style("whitegrid")

        # Define colors for each population's convergence line
        population_colors = {'population_1': 'blue', 'population_2': 'green', 'population_3': 'orange'}

        # Determine global min and max for y-axis limits
        global_min = results_df['hv_value'].min()
        global_max = results_df['hv_value'].max()
        y_margin = 0.05 * (global_max - global_min)
        y_limits = (global_min - y_margin, global_max + y_margin)

        # Loop through the populations and plot each one
        for population, color in population_colors.items():
            subset = results_df[results_df['population'] == population]
            subset['scaled_generation'] = subset['generation'] / 100  # Scale the generations
            
            # Calculate the mean and standard deviation of hypervolume for each scaled generation
            stats = subset.groupby('scaled_generation')['hv_value'].agg(['mean', 'std'])

            # Plot mean hypervolume with the color of the respective population
            ax.plot(stats.index, stats['mean'], linestyle='-', label=f'{population} Mean HV', color=color)
            ax.fill_between(stats.index, stats['mean'] - stats['std'], stats['mean'] + stats['std'], 
                            color=color, alpha=0.2, label=f'{population} ±1 Std Dev')

        # Set plot labels and title
        ax.set_title("Convergence of Three Populations over Generations")
        ax.set_xlabel("Generation (scaled)")
        ax.set_ylabel("Mean Hypervolume (3D)" if results_df['num_dims'].iloc[0] == 3 else "Mean Hyperarea (2D)")
        ax.set_ylim(y_limits)
        ax.legend()
        
        # Display the plot
        plt.tight_layout()
        plt.show()


