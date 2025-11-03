import numpy as np 

# Mean square error 

def MSE(output, target):
    
    return np.sum((target-output)**2/len(target))

# Softmax 

def softmax(output):

    return (np.exp(output))/np.sum(np.exp(output))

# Cross-entropy 



# Test

print(MSE(output = np.array([2,3]), target = np.array([4,6])))

print(softmax(np.array([5,5,7])))





