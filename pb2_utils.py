
import numpy as np
from scipy.optimize import minimize

import GPy
from GPy.kern import Kern
from GPy.core import Param

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances


class TV_SquaredExp(Kern):
    """ Time varying squared exponential kernel.
        For more info see the TV-GP-UCB paper:
        http://proceedings.mlr.press/v51/bogunovic16.pdf
    """

    def __init__(self,
                 input_dim,
                 variance=1.,
                 lengthscale=1.,
                 epsilon=0.,
                 active_dims=None):
        super().__init__(input_dim, active_dims, "time_se")
        self.variance = Param("variance", variance)
        self.lengthscale = Param("lengthscale", lengthscale)
        self.epsilon = Param("epsilon", epsilon)
        self.link_parameters(self.variance, self.lengthscale, self.epsilon)

    def K(self, X, X2):
        # time must be in the far left column
        if self.epsilon > 0.5:  # 0.5
            self.epsilon = 0.5
        if X2 is None:
            X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)
        dists = pairwise_distances(T1, T2, "cityblock")
        timekernel = (1 - self.epsilon)**(0.5 * dists)

        X = X[:, 1:]
        X2 = X2[:, 1:]

        RBF = self.variance * np.exp(
            -np.square(euclidean_distances(X, X2)) / self.lengthscale)

        return RBF * timekernel

    def Kdiag(self, X):
        return self.variance * np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None:
            X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)

        X = X[:, 1:]
        X2 = X2[:, 1:]
        dist2 = np.square(euclidean_distances(X, X2)) / self.lengthscale

        dvar = np.exp(-np.square(
            (euclidean_distances(X, X2)) / self.lengthscale))
        dl = -(2 * euclidean_distances(X, X2)**2 * self.variance *
               np.exp(-dist2)) * self.lengthscale**(-2)
        n = pairwise_distances(T1, T2, "cityblock") / 2
        deps = -n * (1 - self.epsilon)**(n - 1)

        self.variance.gradient = np.sum(dvar * dL_dK)
        self.lengthscale.gradient = np.sum(dl * dL_dK)
        self.epsilon.gradient = np.sum(deps * dL_dK)
        

class TV_MixtureViaSumAndProduct(Kern):
    """ Time varying mixture kernel from CoCaBO:
        http://proceedings.mlr.press/v119/ru20a.html
    """

    def __init__(self,
                 input_dim,
                 variance_1=1.,
                 variance_2=1.,
                 variance_mix=1.,
                 lengthscale=1.,
                 epsilon_1=0.,
                 epsilon_2=0.,
                 mix = 0.5,
                 cat_dims = [],
                 active_dims=None):
        super().__init__(input_dim, active_dims, "time_se")
        
        self.cat_dims = cat_dims
        
        self.variance_1 = Param("variance_1", variance_1)
        self.variance_2 = Param("variance_2", variance_2)
        self.lengthscale = Param("lengthscale", lengthscale)
        self.epsilon_1 = Param("epsilon_1", epsilon_1)
        self.epsilon_2 = Param("epsilon_2", epsilon_2)
        self.mix = Param("mix", mix)
        #self.variance_mix = Param("variance_mix", variance_mix)
        self.variance_mix = variance_mix # fixed
        
        self.link_parameters(self.variance_1, 
                             self.variance_2, 
                             self.lengthscale, 
                             self.epsilon_1,
                             self.epsilon_2,
                             #self.variance_mix,
                             self.mix)
        
    def prepare_data(self, X, X2):
        
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)

        X = X[:, 1:]
        X2 = X2[:, 1:]
        
        # shift becase we have removed time
        cat_dims = [x - 1 for x in self.cat_dims]
        
        X_cat = X[:, cat_dims]
        X_cont = X[:, [x for x in range(X.shape[1]) if x not in cat_dims]]
        
        X2_cat = X2[:, cat_dims]
        X2_cont = X2[:, [x for x in range(X2.shape[1]) if x not in cat_dims]]
        
        return T1, T2, X_cat, X_cont, X2_cat, X2_cont
        
        
    def K1(self, X, X2):
        
        ## format data
        if X2 is None:
            X2 = np.copy(X)
        
        T1, T2, X_cat, X_cont, X2_cat, X2_cont = self.prepare_data(X, X2)        

        ## time kernel k_t
        dists = pairwise_distances(T1, T2, "cityblock")
        timekernel_1 = (1 - self.epsilon_1)**(0.5 * dists)
        
        ## SE kernel k_se
        RBF = self.variance_1 * np.exp(
            -np.square(euclidean_distances(X_cont, X2_cont)) / self.lengthscale)
        
        ## k1 = k_se * k_t
        k1 = RBF * timekernel_1

        return k1        
    
    def K2(self, X, X2):
        
        ## format data
        if X2 is None:
            X2 = np.copy(X)
        
        T1, T2, X_cat, X_cont, X2_cat, X2_cont = self.prepare_data(X, X2) 

        ## time kernel k_t
        dists = pairwise_distances(T1, T2, "cityblock")
        timekernel_2 = (1 - self.epsilon_2)**(0.5 * dists) 
        
        ## CategoryOverlapKernel
        # convert cat to int so we can subtract
        cat_vals = list(set(X_cat.flatten()).union(set(X2_cat.flatten())))
        for i, val in enumerate(cat_vals):
            X_cat = np.where(X_cat==val, i, X_cat) 
            X2_cat = np.where(X2_cat==val, i, X2_cat) 
        diff = X_cat[:, None] - X2_cat[None, :]
        diff[np.where(np.abs(diff))] = 1
        diff1 = np.logical_not(diff)
        k_cat = self.variance_2 * np.sum(diff1, -1) / len(self.cat_dims)
        
        ## k2 = k_cat * k_t
        k2 = k_cat * timekernel_2

        return k2                

    def K(self, X, X2):
        
        ## clip epsilons
        if self.epsilon_1 > 0.5:  # 0.5
            self.epsilon_1 = 0.5
            
        if self.epsilon_2 > 0.5:  # 0.5
            self.epsilon_2 = 0.5
            
        ## format data
        if X2 is None:
            X2 = np.copy(X)
            
        k1 = self.K1(X, X2)
        k2 = self.K2(X, X2)
                
        ##### K_mix
        k_out = self.variance_mix * ((1 - self.mix) * 0.5 * (k1 + k2)
                                + self.mix * k1 * k2)

        return k_out

    def Kdiag(self, X):
        """
        Not sure what this is for?

        """
        return np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        ## format data

        if X2 is None:
            X2 = np.copy(X)
            
        k1_xx = self.K1(X, X2)
        k2_xx = self.K2(X, X2)
        
        K_mix = self.K(X, X2)

        T1, T2, X_cat, X_cont, X2_cat, X2_cont = self.prepare_data(X, X2) 
        
        # compute common terms before K1 grads and K2 grads
        n = pairwise_distances(T1, T2, "cityblock") / 2
        k_t1= (1-self.epsilon_1)**(n-1)
        k_t2= (1-self.epsilon_2)**(n-1)

        k_x=self.variance_1 * np.exp(-np.square(euclidean_distances(X_cont, X2_cont)) / self.lengthscale)
        
        # convert cat to int so we can subtract
        cat_vals = list(set(X_cat.flatten()).union(set(X2_cat.flatten())))
        for i, val in enumerate(cat_vals):
            X_cat = np.where(X_cat==val, i, X_cat) 
            X2_cat = np.where(X2_cat==val, i, X2_cat) 
        diff = X_cat[:, None] - X2_cat[None, :]
        diff[np.where(np.abs(diff))] = 1
        diff1 = np.logical_not(diff)
        k_h = self.variance_2 * np.sum(diff1, -1) / len(self.cat_dims)


        #### K1 grads
        dist2 = np.square(euclidean_distances(X_cont, X2_cont)) / self.lengthscale

        dvar1 = np.exp(-np.square(
            (euclidean_distances(X_cont, X2_cont)) / self.lengthscale))

        
        dl = -(euclidean_distances(X_cont, X2_cont)**2 * self.variance_1 *
               np.exp(-dist2)) * self.lengthscale**(-2)

        
        deps1 = -n * (1 - self.epsilon_1)**(n - 1)
        dKout_l = (1-self.mix) * k_t1*dl + self.mix* self.K2(X,X2) * k_t1*dl
        dKout_var1 = (1-self.mix) * k_t1 * dvar1 + self.mix * self.K2(X,X2)* k_t1 * dvar1 
        dKout_eps1= (1-self.mix)*k_x*deps1 + self.mix*self.K1(X,X2)*k_x*deps1


        self.variance_1.gradient = np.sum(dKout_var1 * dL_dK)
        self.lengthscale.gradient = np.sum(dKout_l * dL_dK)
        self.epsilon_1.gradient = np.sum(dKout_eps1 * dL_dK)
        
        #### K2 grads
        dvar2 = np.sum(diff1, -1) / len(self.cat_dims)
        
        deps2 = -n * (1 - self.epsilon_2)**(n - 1)
        dKout_var2=(1-self.mix)*	k_t2*dvar2 + self.mix*self.K1(X,X2)* k_t2*dvar2 
        dKout_eps2=(1-self.mix)*k_h*deps2 + self.mix*self.K2(X,X2)*k_h*deps2


        self.variance_2.gradient = np.sum(dKout_var2 * dL_dK)
        self.epsilon_2.gradient = np.sum(dKout_eps2 * dL_dK)
        
        #### K_mix grads
        
        self.mix.gradient = np.sum(dL_dK *( -(k1_xx + k2_xx) + (k1_xx * k2_xx)))
        
        #self.variance_mix.gradient = \
        #    np.sum(K_mix * dL_dK) / self.variance_mix
        
        


def normalize(data, wrt):
    """ Normalize data to be in range (0,1), with respect to (wrt) boundaries,
        which can be specified.
    """
    return (data - np.min(wrt, axis=0)) / (
        np.max(wrt, axis=0) - np.min(wrt, axis=0))
    

def standardize(data):
    """ Standardize to be Gaussian N(0,1). Clip final values.
    """
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    return np.clip(data, -2, 2)


def UCB(m, m1, x, fixed, kappa=0.5):
    """ UCB acquisition function. Interesting points to note:
        1) We concat with the fixed points, because we are not optimizing wrt
           these. This is the Reward and Time, which we can't change. We want
           to find the best hyperparameters *given* the reward and time.
        2) We use m to get the mean and m1 to get the variance. If we already
           have trials running, then m1 contains this information. This reduces
           the variance at points currently running, even if we don't have
           their label.
           Ref: https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf

    """

    c1 = 0.2
    c2 = 0.4
    beta_t = np.max([c1 * np.log(c2 * m.X.shape[0]), 0])
    kappa = np.sqrt(beta_t)

    xtest = np.concatenate((fixed.reshape(-1, 1), np.array(x).reshape(-1,
                                                                      1))).T

    try:
        preds = m.predict(xtest)
        preds = m.predict(xtest)
        mean = preds[0][0][0]
    except ValueError:
        mean = -9999

    try:
        preds = m1.predict(xtest)
        var = preds[1][0][0]
    except ValueError:
        var = 0
    return mean + kappa * var


def optimize_acq(func, m, m1, fixed, num_f):
    """ Optimize acquisition function."""

    opts = {"maxiter": 200, "maxfun": 200, "disp": False}

    T = 10
    best_value = -999
    best_theta = m1.X[0, :]

    bounds = [(0, 1) for _ in range(m.X.shape[1] - num_f)]

    for ii in range(T):
        x0 = np.random.uniform(0, 1, m.X.shape[1] - num_f)

        res = minimize(
            lambda x: -func(m, m1, x, fixed),
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=opts)

        val = func(m, m1, res.x, fixed)
        if val > best_value:
            best_value = val
            best_theta = res.x

    return (np.clip(best_theta, 0, 1))


def select_length(Xraw, yraw, bounds, num_f):
    """Select the number of datapoints to keep, using cross validation
    """
    min_len = 200

    if Xraw.shape[0] < min_len:
        return (Xraw.shape[0])
    else:
        length = min_len - 10
        scores = []
        while length + 10 <= Xraw.shape[0]:
            length += 10

            base_vals = np.array(list(bounds.values())).T
            X_len = Xraw[-length:, :]
            y_len = yraw[-length:]
            oldpoints = X_len[:, :num_f]
            old_lims = np.concatenate((np.max(oldpoints, axis=0),
                                       np.min(oldpoints, axis=0))).reshape(
                                           2, oldpoints.shape[1])
            limits = np.concatenate((old_lims, base_vals), axis=1)

            X = normalize(X_len, limits)
            y = standardize(y_len).reshape(y_len.size, 1)

            kernel = TV_SquaredExp(
                input_dim=X.shape[1], variance=1., lengthscale=1., epsilon=0.1)
            m = GPy.models.GPRegression(X, y, kernel)
            m.optimize(messages=True)

            scores.append(m.log_likelihood())
        idx = np.argmax(scores)
        length = (idx + int((min_len / 10))) * 10
        return (length)