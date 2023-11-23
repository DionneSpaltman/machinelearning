import time
from sys import argv
from machinelearning.baseline import baseline
from machinelearning.baseline_improved import baseline_improved
from machinelearning.randomforest_chatgpt import randomforest_chatgpt
from machinelearning.perceptron import perceptron
from machinelearning.splitdata20_80 import splitdata20_80

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
    
    else:
        print(f"Invalid algorithm: {argv[1]}. Please choose 'baseline' or 'improved_baseline'.")