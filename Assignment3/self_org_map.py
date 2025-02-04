import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pycountry
import pypopulation
import math


class SelfOrganizingMap():
    def __init__(self, file_name, learning_rate, iterations):
        self.file_name = file_name
        self.process_dataset()
        self.data = self.data_frame.values.tolist()
        self.column_attributes = len(self.data[0])
        self.grid_size = 10
        self.num_clusters = self.grid_size * self.grid_size
        self.weights = np.random.randint(256, size=(self.grid_size, self.grid_size, 3)) / 255.0
        self.learning_rate = learning_rate
        self.radius = self.grid_size // 2
        self.time_constant = iterations / math.log(self.radius)
        self.weight_changes = []  
        self.country_color = []

    def process_dataset(self):

        self.data_frame = pd.read_csv(self.file_name)        
        self.data_frame = self.data_frame.groupby(['Country_Region'])[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
        self.unique_country_names = self.data_frame['Country_Region'].values.tolist()
        data_frame_copy = self.data_frame.copy()
        # data_frame_copy.to_csv('data.csv', index=False)

        #population added to the dataset and data is normalized in accordance to the population to see the affect of
        #each of the 3 params - Confirmed, Deaths, Recovered in accordance with each country's population
        #this has been done to see the affect of covid 19 on each country based on their population
        #comment/uncomment the section if you don't want/want to do this - see the effect with respect to the population
        #and instead directly use the dataset

        #section starts
        rows_to_drop = []
        pop_attr = []
        for index, row in data_frame_copy.iterrows():
            if pycountry.countries.get(name=row['Country_Region']) is not None:
                pop_attr.append(pypopulation.get_population_a3(pycountry.countries.get(name=row['Country_Region']).alpha_3))
            else:
                rows_to_drop.append(index)
        data_frame_copy = data_frame_copy.drop(rows_to_drop)
        data_frame_copy['Population'] = pop_attr
        data_frame_copy['Confirmed'] = (data_frame_copy['Confirmed']) / (data_frame_copy['Population'])
        data_frame_copy['Deaths'] = (data_frame_copy['Deaths']) / (data_frame_copy['Population'])
        data_frame_copy['Recovered'] = (data_frame_copy['Recovered']) / (data_frame_copy['Population'])
        data_frame_copy.drop(['Population'], axis=1, inplace=True)
        #section ends

        data_frame_copy.drop(['Country_Region'], axis=1, inplace=True)
        self.data_frame = data_frame_copy
        #filling of the NaN values
        columns_with_nan = self.data_frame.columns[self.data_frame.isna().any()].tolist()
        for column_name in columns_with_nan:
            avg = self.data_frame[column_name].mean()
            self.data_frame[column_name] = self.data_frame[column_name].fillna(avg)
        #normalization of the data
        scaler = MinMaxScaler()
        data_frame_copy = self.data_frame.copy()
        self.data_frame = pd.DataFrame(scaler.fit_transform(data_frame_copy), columns=data_frame_copy.columns)
        # self.data_frame.to_csv('data.csv', index=False)

    def choose_cluster(self, attrs):
        min_index = np.unravel_index(np.argmin(np.sum((np.array(self.weights) - np.array(attrs)) ** 2, axis=2)),(self.grid_size, self.grid_size))
        return list(min_index)

    def calculate_distance(self, row, col, target_row, target_col):
        return ((row - target_row)**2 + (col - target_col)**2)**0.5

    def get_learning_effect(self, distance):
        if distance <= self.radius:
            return self.learning_rate * np.exp(-(distance**2) / (2 * (self.radius**2)))
        else:
            return 0

    def update_weights_return_diff(self, attrs, best_cluster_index):
        row_best_cluster, column_best_cluster = best_cluster_index
        weight_diff = 0
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                distance = self.calculate_distance(row, column, row_best_cluster, column_best_cluster)
                learning_effect = self.get_learning_effect(distance)
                if learning_effect > 0: 
                    for weight_index, input_val in enumerate(attrs):
                        old_weight = self.weights[row, column, weight_index]
                        self.weights[row, column, weight_index] += learning_effect * (input_val - old_weight)
                        weight_diff += abs(self.weights[row, column, weight_index] - old_weight)
        return weight_diff

    def assign_color_from_weights(self):
        figure = plt.figure(figsize=(10, 6))
        ax = figure.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.grid_size))
        ax.set_ylim((0, self.grid_size))
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                color = (self.weights[row][column][:3])
                ax.add_patch(plt.Rectangle((row, column), 1, 1, facecolor=color, edgecolor='none'))
                name_label = ''
                for color_lst in self.country_color:
                    if color_lst[1] == [row, column]:
                        color_lst.append(color)
                        name_label = name_label + color_lst[0] + "\n"
                ax.text(row + 0.5, column + 0.5, name_label, ha='center', va='center', color="white", fontsize=5)
        plt.show()

    def get_world_map(self):
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.set_facecolor('azure')
        world.plot(ax=ax, color='white', edgecolor='black')
        map_countries = world['name'].tolist()
        # print(len(map_countries))
        for color_lst in self.country_color:
            country_name = color_lst[0]
            color = color_lst[2]  
            if country_name in map_countries:
                world[world.name == country_name].plot(color=color, ax=ax, linewidth=0.3, edgecolor='maroon')
                map_countries.remove(country_name)
            else:
                print(country_name + " not found in map")
        # print(len(map_countries))
        # print(sorted(map_countries))        
        ax.set_title('World Map Showing COVID-19 (SOM) Clustering Results', fontsize=25, fontweight='bold')
        plt.show()


def self_organizing_maps(file_name, learning_rate, iterations):
    som = SelfOrganizingMap(file_name, learning_rate, iterations)
    som.country_color = [[country_name, som.choose_cluster(x)] for country_name, x in zip(som.unique_country_names, som.data)]
    som.assign_color_from_weights()
    for iteration in range(1, iterations + 1):
        total_weight_diff = 0
        for x in som.data:
            best_matching_index = som.choose_cluster(x)
            total_weight_diff += som.update_weights_return_diff(x, best_matching_index)
        som.weight_changes.append(total_weight_diff)
        som.learning_rate *= math.exp(-(iteration / som.time_constant))
        som.radius *= math.exp(-(iteration / som.time_constant))
        print(f"Iteration {iteration}: Total weight change: {total_weight_diff}")
        if(total_weight_diff < 0.00001):
            break
        if iteration % 10 == 0:
            som.country_color = [[country_name, som.choose_cluster(x)] for country_name, x in zip(som.unique_country_names, som.data)]
            som.assign_color_from_weights()
    som.country_color = [[country_name, som.choose_cluster(x)] for country_name, x in zip(som.unique_country_names, som.data)]
    som.assign_color_from_weights()
    plot_weight_changes(som.weight_changes)
    som.get_world_map()

def plot_weight_changes(weight_changes):
    plt.figure(figsize=(10, 6))
    plt.plot(weight_changes, marker='o')
    plt.title("Weight Changes per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Total Weight Change")
    plt.grid(True)
    plt.show()

def main():
    # I have cleaned the data of the provided input file by renaming some countries to match the names in the world map
    # according to the Geopandas library
    filename = 'Q1_countrydata_Cleaned.csv'
    learning_rate = 0.1
    # learning_rate = 0.2
    
    #This won't be running for 500 iterations because as we are decaying exponenetially with the no of iterations
    #At the point the difference would become zero, it would break and the map would form.
    iterations = 500
    self_organizing_maps(filename, learning_rate, iterations)

if __name__ == '__main__':
    main()