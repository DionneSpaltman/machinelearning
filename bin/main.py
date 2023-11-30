import time
from sys import argv
import pandas as pd
from baseline import baseline
from baseline_improved import baseline_improved
from randomforest_chatgpt import randomforest_chatgpt
from perceptron import perceptron
from splitdata20_80 import splitdata20_80
from bin.preprocessing_copy import year_column, entrytype_column, editor_column, publisher_column, author_column, title_abstract, preparing_data

def run_baseline():
    start_time = time.time()
    baseline()
    end_time = time.time()
    print(f"Baseline algorithm took {end_time - start_time:.3f} seconds.")

def run_improved_baseline(algorithm):
    start_time = time.time()
    baseline_improved(algorithm)
    end_time = time.time()
    print(f"Improved baseline algorithm took {end_time - start_time:.3f} seconds.")

def run_randomforest_chatgpt():
    start_time = time.time()
    randomforest_chatgpt()
    end_time = time.time()
    print(f"Random forest (chat gpt) algorithm took {end_time - start_time:.3f} seconds.")

def run_perceptron():
    start_time = time.time()
    perceptron()
    end_time = time.time()
    print(f"Perceptron algorithm took {end_time - start_time:.3f} seconds.")

def run_splitdata():
    start_time = time.time()
    splitdata20_80()
    end_time = time.time()
    print(f"The split data 20/80 algorithm took {end_time - start_time:.3f} seconds.")

if __name__ == '__main__':
    # Code to ask the user for the algorithm
    if len(argv) == 1:
        print("Usage: python3 main.py algorithm")
        exit(1)

    # Running the selected algorithm
    if argv[1] == "baseline":
        run_baseline()
    elif argv[1] == "improved_baseline":
        if argv[2] == "DecisionTreeRegressor":
            run_improved_baseline("DecisionTreeRegressor")
        elif argv[2] == "RandomForestRegressor":
            run_improved_baseline("RandomForestRegressor")
        else: 
            run_improved_baseline("Ridge")
    elif argv[1] == "randomforest_chatgpt":
        run_randomforest_chatgpt()
    elif argv[1] == "perceptron":
        run_perceptron()
    elif argv[1] == "splitdata":
        splitdata20_80()
    
    elif argv[1] == "setup":
        data = pd.read_json('input/train.json')
        year_column(data)
        entrytype_column(data)
        editor_column(data)
        publisher_column(data)
        author_column(data)
        title_tfidf_df, abstract_tfidf_df = title_abstract(data)
        X_train, X_test, y_train, y_test = preparing_data(data, title_tfidf_df, abstract_tfidf_df)
        
    else:
        print(f"Invalid algorithm: {argv[1]}. Please choose 'baseline' or 'improved_baseline'.")