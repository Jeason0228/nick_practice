from matplotlib.figure import Figure
import numpy as np
from numpy.lib.function_base import append


class HMM:
    """Ref: https://zhuanlan.zhihu.com/p/85454896
    """
    def __init__(self, S, F, SD, OD, TD) -> None:
        self.samples = S
        self.features = F
        self.samples_dist = SD
        self.ob_dist = OD
        self.trans_dist = TD
    
    def random_choice(self, dist):
        return np.random.choice(np.arange(len(dist)), p=dist)
    
    def run(self, T):
        observed = []
        sample = self.random_choice(self.samples_dist)
        first_ob = self.random_choice(self.ob_dist[sample])
        observed.append(first_ob)
        for _ in range(T-1):
            sample = self.random_choice(self.trans_dist[sample])
            ob = self.random_choice(self.ob_dist[sample])
            observed.append(ob)
        return observed
    
    def forward(self, X):
        alpha = self.samples_dist * self.ob_dist[:, X[0]]
        
        ## based on standard equation
        # alpha_next = np.empty(self.N)
        # for j in range(self.N):
        #     alpha_next[j] = np.sum(self.A[:,j] * alpha * self.B[j,x])
        # alpha = alpha_next
        
        # 矩阵化
        for x in X[1:]:
            alpha = np.sum(self.trans_dist * alpha.reshape(-1, 1) * self.ob_dist[:, x], axis=0)
        return alpha.sum()
    
    def backward(self, X):
        beta = np.ones(self.samples)
        
        ## based on standard equation
        # for x in X[:0:-1]:
        #     beta_next = np.empty(self.samples)
        #     for j in range(self.samples):
        #         beta_next[j] = np.sum(self.trans_dist[j,:] * beta * self.ob_dist[:, x])
        #     beta = beta_next
        
        # 矩阵化
        for x in X[:0:-1]:
            beta = np.sum(self.trans_dist * beta * self.ob_dist[:, x], axis=1)
        return np.sum(beta * self.samples_dist * self.ob_dist[:, X[0]])

if __name__ == '__main__':
    pi = np.array([.25, .25, .25, .25])
    A = np.array([
        [0,  1,  0, 0],
        [.4, 0, .6, 0],
        [0, .4, 0, .6],
        [0, 0, .5, .5]])
    B = np.array([
        [.5, .5],
        [.3, .7],
        [.6, .4],
        [.8, .2]])
    hmm = HMM(4, 2, pi, B, A)
    X = hmm.run(5)
    X = [0, 0, 1, 1, 0]
    fore_prob = hmm.forward(X)
    back_prob = hmm.backward(X)
    print(f'For {X}, the probability is {fore_prob, back_prob}')
    