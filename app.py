import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def filter_data(data):
    filtered_data = data[['Language', 'Response_Accuracy']].dropna()
    languages = filtered_data['Language'].tolist()
    accuracies = filtered_data['Response_Accuracy'].astype(float).tolist()
    return languages, accuracies

def encode_languages(languages):
    unique_languages = list(set(languages))
    language_map = {lang: idx for idx, lang in enumerate(unique_languages)}
    encoded_languages = [language_map[lang] for lang in languages]
    return encoded_languages, language_map

def linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_mean = np.mean(np.multiply(x, y))
    xx_mean = np.mean(np.multiply(x, x))

    slope = (xy_mean - x_mean * y_mean) / (xx_mean - x_mean * x_mean)
    intercept = y_mean - slope * x_mean
    return slope, intercept

def plot_regression(x, y, slope, intercept, save_path):
    plt.scatter(x, y, color='blue')
    plt.plot(x, slope * np.array(x) + intercept, color='red')
    plt.xlabel('Encoded Language')
    plt.ylabel('Response Accuracy')
    plt.title('Linear Regression: Language vs Response Accuracy')
    plt.savefig(f'{save_path}/regression_plot.png')
    plt.close()
    print(f"Graph saved as '{save_path}/regression_plot.png'.")

def main():
    filepath = './archive/deepseek_vs_chatgpt.csv'
    save_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    data = load_data(filepath)
    languages, accuracies = filter_data(data)
    encoded_languages, language_map = encode_languages(languages)
    
    slope, intercept = linear_regression(encoded_languages, accuracies)
    print(f'Regression Line: y = {slope}x + {intercept}')
    
    plot_regression(encoded_languages, accuracies, slope, intercept, save_path)

if __name__ == "__main__":
    main()
