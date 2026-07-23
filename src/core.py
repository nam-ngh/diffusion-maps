import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

class Map:
    @staticmethod
    def compute_markov(data, epsilon=1.0, alpha=1.0):
        # pairwise distances
        pair_dist = squareform(pdist(data))
        
        # kernel matrix K 
        K_raw = np.exp(-(pair_dist ** 2)/epsilon)

        # degree vec
        q = np.sum(K_raw, axis=1)

        # normalised kernel & degree vec
        K = K_raw / (np.outer(q ** alpha, q ** alpha))
        d = np.sum(K, axis=1)

        # transition matrix
        P_sym = K / (np.sqrt(d)[:, None]) / (np.sqrt(d)[None, :])
        return P_sym, d

    @staticmethod
    def qr_mgs(A, min_norm=1e-15):
        '''QR decompose any matrix A with MGS'''
        A = A.copy()
        rows, cols = A.shape
        Q = np.zeros((rows, cols))
        R = np.zeros((cols, cols))

        for v1 in range(cols):
            v1_norm = np.linalg.norm(A[:, v1])
            if v1_norm < min_norm:
                continue
            R[v1, v1] = v1_norm
            Q[:, v1] = A[:, v1] / v1_norm
            for v2 in range(v1 + 1, cols):
                projection = np.dot(Q[:, v1], A[:, v2])
                R[v1, v2] = projection
                A[:, v2] = A[:, v2] - projection * Q[:, v1]

        return Q, R
    
    @staticmethod
    def eigen_decompose(A, iter_max: int = 1000, tol: float = 1e-10):
        '''
        Compute eigenvals & eigenvecs of square mat. A with iterative application of QR
        Returns eigenvals (n) and eigenvecs (n,n)
        '''
        A = A.copy()
        for i in range(iter_max):
            Q, R = Map.qr_mgs(A)
            A = R @ Q
            if i == 0:
                V = Q
            else:
                V = V @ Q

            off_diag_norm = np.linalg.norm(A - np.diag(np.diag(A)))
            if off_diag_norm < tol:
                break
        
        # sorting
        eigenvals = np.diag(A)
        sort_idx = np.argsort(eigenvals)
        eigenvals_sorted = eigenvals[sort_idx]
        eigenvecs_sorted = V[:, sort_idx]
        
        return eigenvals_sorted, eigenvecs_sorted
    
    def plot(self, data, eigenvals, eigenvecs):
        n_vecs = 5
        max_dims = eigenvecs.shape[1]
        top_idx = [max_dims - k - 21 for k in range(n_vecs)]
        
        # 1. Eigenvalue spectrum
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(eigenvals, 'o-')
        ax.set_xlabel('Index'); ax.set_ylabel('Eigenvalue')
        ax.set_title('Eigenvalue spectrum')
        plt.tight_layout()
        plt.show()

        # 2. Original 3D cloud coloured by each eigenvector
        fig = plt.figure(figsize=(5 * n_vecs, 4))
        for k in range(n_vecs):
            psi = eigenvecs[:, top_idx[k]]
            ax = fig.add_subplot(1, n_vecs, k + 1, projection='3d')
            sc = ax.scatter(
                data[:, 0], data[:, 1], data[:, 2],
                c=psi, cmap='RdBu', s=5
            )
            plt.colorbar(sc, ax=ax, shrink=0.6)
            ax.set_title(f'psi{top_idx[k]}  (lamb={eigenvals[top_idx[k]]:.3f})')
        plt.tight_layout()
        plt.show()

        # 3. Diffusion embedding:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            eigenvecs[:, top_idx[0]], eigenvecs[:, top_idx[1]], eigenvecs[:, top_idx[2]],
            c=eigenvecs[:, 1], cmap='viridis', s=5
        )
        plt.colorbar(sc, ax=ax, shrink=0.6)
        ax.set_xlabel('psi1'); ax.set_ylabel('psi2'); ax.set_zlabel('psi3')
        ax.set_title('Diffusion embedding')
        plt.tight_layout()
        plt.show()

    def run(self, data, plot=False):
        P_sym, d = self.compute_markov(data)
        # eigenvals, eigenvecs_sym = self.eigen_decompose(P_sym, iter_max=100)
        eigenvals, eigenvecs_sym = np.linalg.eigh(P_sym)

        # convert back to eigenvectors of P
        d_inv_sqrt = 1.0 / np.sqrt(d)
        eigenvecs = eigenvecs_sym * d_inv_sqrt[:, np.newaxis]

        
        if plot:
            # renormalise
            # l2_norm = np.linalg.norm(eigenvecs, axis=0, keepdims=True)
            norm = np.sqrt(np.sum(d[:, None] * eigenvecs**2, axis=0))
            eigenvecs = eigenvecs / norm
            self.plot(data, eigenvals, eigenvecs)
        else:
            print(f'Eigenvalues: {eigenvals.shape}, Eigenvectors: {eigenvecs.shape}')
            return eigenvals, eigenvecs

        # check orthogonality
        # print(eigenvecs.T[202] @ eigenvecs.T[100])

if __name__ == "__main__":
    data = np.load('data/population.npy')
    print(data[:3])
    map = Map()
    map.run(data, plot=True)