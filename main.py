# import time 
# from sys import argv
# import re

# # import algorithms
# from machinelearning.baseline import baseline
# from baseline_improved import baseline_improved
  
# if __name__ == '__main__':
#     # Code to ask user for algorithm   
#     if len(argv) == 1:
#         print("Usage: python3 main.py algorithm")
#         exit(1)

# # -------------------------------------------------Running the Algorithms-------------------------------------------------#
            
#     if len(argv) > 2:   
#         # Running the baseline algorithm 
#         if argv[2] == "baseline":
#             start_time = time.time()
#             baseline()
#             end_time = time.time()
#             print(f"It took {end_time - start_time:.3f} seconds.")
            

#         # Running the improved baseline algorithm 
#         elif argv[2] == "improved_baseline":
#             baseline_improved()
        

# # -------------------------------------------------Visualisation & Experiments-------------------------------------------------#
        
#         # elif argv[2] == "improved_baseline":
        


import time
from sys import argv
from machinelearning.baseline import baseline
from baseline_improved import baseline_improved

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
    
    else:
        print(f"Invalid algorithm: {argv[1]}. Please choose 'baseline' or 'improved_baseline'.")