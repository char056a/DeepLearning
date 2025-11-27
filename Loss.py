import numpy as np 

# Mean square error 
def MSE(output, target):
    return np.sum((target-output)**2/len(target))

# Softmax for a single vector
def softmax(output):
    return (np.exp(output))/np.sum(np.exp(output))

# Cross-entropy (only for one training example)
def cross_entropy_single(correct_onehot, network_output):
    S = softmax(network_output)
    S = np.sum(S * correct_onehot)
    S = -np.log(S + 1e-12)
    return S 

# Softmax for a batch 
def softmax_matrix(output):
    shifted = output - np.max(output, axis=0, keepdims=True) 
    exp_vals = np.exp(shifted) 
    return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)

# Cross-entropy for a batch
def cross_entropy_batch(correct_onehot, network_output):
    S = softmax_matrix(network_output)
    S = np.sum(S * correct_onehot, axis=0)
    S = -np.log(S + 1e-12)
    return np.mean(S) # mean over the whole batch

# Test
#print("MSE:")
#print(MSE(output = np.array([2,3]), target = np.array([4,6])))
#print("Softmax:")
#print(softmax(np.array([1,2,20,3,4])))
#print("Cross:")
#print(cross_entropy_single(np.array([0,0,1,0,0]),[1,2,20,3,4]))
#print("Softmax_matrix: small numbers")
#print(softmax_matrix(np.array([1, 2, 3]))) 
#print("Softmax_matrix: large numbers") 
#print(softmax_matrix(np.array([1000, 1001, 999]))) 



