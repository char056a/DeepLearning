import numpy as np 

# Mean square error 

def MSE(output, target):
    
    return np.sum((target-output)**2/len(target))

# Softmax 

def softmax(output):

    return (np.exp(output))/np.sum(np.exp(output))

# Cross-entropy (only for one training example)

def cross_entropy_single(correct_onehot, network_output):

    S = softmax(network_output)
    S = np.sum(S * correct_onehot)
    S = -np.log(S)
    return S 

# Test
print("MSE:")
print(MSE(output = np.array([2,3]), target = np.array([4,6])))
print("Softmax:")
print(softmax(np.array([1,2,20,3,4])))
print("Cross:")
print(cross_entropy_single(np.array([0,0,1,0,0]),[1,2,20,3,4]))




