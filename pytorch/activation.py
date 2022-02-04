import numpy as np
import matplotlib.pyplot as plt

def activation(type):
    if type == 'tanh':
        return lambda z:(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    else:
        print('No implementation')


def show(func): 
    start = -10
    end = 10
    step = (end-start) / 0.01
    x = np.linspace(start, end, step)
    y = func(x)
    
    fig = plt.figure(1)
    plt.plot(x, y, label='tanh')
    plt.grid(True)
    plt.legend()
    plt.show(fig)
    

if __name__ == '__main__':
    show(activation('tanh'))
    
    