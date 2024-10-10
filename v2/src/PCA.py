
import numpy as np
import jax
import jax.numpy as jnp

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.S = None
        self.V_r = None
        self.Z = None
        self.m = None
        
    def fit(self, X):
        X = X - jnp.mean(X, axis=0)
        self.m = X.shape[0]
        U, S, V = jnp.linalg.svd(X, full_matrices=False)
        self.S = S
        self.V_r = V[:self.n_components, :]
        self.Z = jnp.dot(X, self.V_r.T)
        
    def transform(self, X):
        return jnp.dot(X, self.V_r.T)
    
    def get_eigenvectors(self):
        return self.V_r
    
    def get_eigenvalues(self):
        return self.S ** 2 / (self.m - 1)
    
    def reconstruct(self, Z):
        return jnp.dot(Z, self.V_r)

# seems to be working
class KPCA:
    def __init__(self, n_components, kernel):
        '''
        kernel: callable - the dot product in the feature space
        '''
        self.n_components = n_components
        self.kernel = kernel
        self.eigenvectors = None
        self.eigenvalues = None
        self.m = None
        self.K = None
        
    def fit(self, X):
        self.m = X.shape[0]
        vectorized_kernel = jax.vmap(lambda x: jax.vmap(lambda y: self.kernel(x, y))(X))

        K = vectorized_kernel(X)
        one_m = jnp.ones((self.m, self.m)) / self.m
        K = K - jnp.dot(one_m, K) - jnp.dot(K, one_m) + jnp.dot(one_m, jnp.dot(K, one_m))
        self.K = K
        self.eigenvalues, self.eigenvectors = jnp.linalg.eigh(K)
        
        idx = jnp.argsort(self.eigenvalues)[::-1]
        
        self.eigenvectors = self.eigenvectors[:, idx]
        self.eigenvalues = self.eigenvalues[idx]
        
        self.V_r = self.eigenvectors[:, :self.n_components]
        
    def transform(self, idx):
        return jnp.dot(self.K[idx], self.V_r)
    
    
def test_KPCA():
    def gaussian_kernel(x, y, sigma=1):
        return jnp.exp(-jnp.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
    
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.2)
    
    kpca = KPCA(n_components=3, kernel=gaussian_kernel)    
    # transformed = kpca.transform(X)
    kpca.fit(X)
    transformed = kpca.transform(jnp.arange(100))
    print(transformed.shape)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=y, cmap='viridis')
    ax.set_title('3D Scatter plot of KPCA transformed data')
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt    
    test_KPCA()

    # test_kernel_PCA()