def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
