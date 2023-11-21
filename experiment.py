import time 
from baseline_improved import baseline_improved

def run(algorithm, ):
    """
    Input: algorithm, 
    """
    start_time = time.time()

    baseline_improved()
    
    
    end_time = time.time()
    print(f"It took {end_time - start_time:.3f} seconds.")