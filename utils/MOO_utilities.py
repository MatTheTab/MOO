import os
import random
import matplotlib.pyplot as plt

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
            plt.xticks(range(0, len(values), 10))
        else:
            plt.xlim([0, val_range])
            plt.xticks(range(0, val_range, 10))
        

        plt.grid(True)
        plt.show()

class RegressionModelsCombined():
    def __init__(self, data, window_size, test_size = 0.2):
        self.data = data
        self.windows_size = window_size
        self.test_size = test_size

    def create_X_y(self):
        self.X = []
        self.y = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        all_data = []
        
        for company in self.data.keys():
            company_data = self.data[company]
            for val in company_data:
                all_data.append(val)
        
        for i in range(self.window_size, len(all_data)):
            self.X.append(all_data[i - self.window_size:i])
            self.y.append(all_data[i])

        self.X_train = self.X[:len(self.X)*(1-self.test_size)].copy()
        self.y_train = self.y[:len(self.X)*(1-self.test_size)].copy()
        self.X_test = self.X[len(self.X)*(1-self.test_size):].copy()
        self.y_test = self.y[len(self.X)*(1-self.test_size):].copy()

    
    def train_linear():
        pass

    def train_ARIMA():
        pass
    
    def train_XGBOOST():
        pass

    def train_NODE():
        pass

class Solver():
    def __init__(self, data, predicted_vals, risks):
        self.data = data
        self.predicted_vals = predicted_vals
        self.risks = risks

    def plot_predictions(self, company_name=None):
        if company_name is None:
            company_name = random.choice(list(self.data.keys()))

        if company_name not in self.data or company_name not in self.predicted_vals:
            print(f"Company '{company_name}' not found in data or predictions.")
            return
        
        actual_values = self.data[company_name]
        predicted_values = self.predicted_vals[company_name]
        actual_timesteps = list(range(len(actual_values)))
        predicted_timesteps = list(range(len(actual_values), len(actual_values) + len(predicted_values)))
        
        plt.figure(figsize=(10, 5))
        plt.plot(actual_timesteps, actual_values, linestyle='-', markersize=3, label='Actual', color='blue')
        plt.plot(predicted_timesteps, predicted_values, linestyle='--', markersize=3, label='Predicted', color='red')
        
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.title(f"Actual vs Predicted - {company_name}")
        plt.legend()
        plt.grid(True)
        plt.show()