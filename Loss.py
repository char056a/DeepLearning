import numpy as np 

# Mean square error 

def MSE(output, target):
    
    return np.sum((target-output)**2/len(target))

# Cross-entropy 





# Test

print(MSE(output = np.array([2,3]), target = np.array([4,6])))





