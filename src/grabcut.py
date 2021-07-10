import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from igraph import Graph
import tqdm.auto as tqdm
import matplotlib.pyplot as plt

class GrabCut:
    '''
    Grabcut Class. Has all the functions to perform Grabcut
    '''
    
    BGD = 0
    FGD = 1
    PR_BGD = 2
    PR_FGD = 3
    should_fit = True

    def __init__(self, beta, gamma, num_iters=5):
        self.num_iters = num_iters
        self.beta = beta
        self.gamma = gamma


    def calcNweights(self, img):
        '''
        Calculates the Pairwise potentials (or the Smoothness term)
        '''
        
        img = img.astype(int)

        leftW = np.zeros(img.shape[:-1])
        upleftW = np.zeros(img.shape[:-1])
        upW = np.zeros(img.shape[:-1])
        uprightW = np.zeros(img.shape[:-1])

        leftW[:, 1:] = self.gamma * np.exp(-self.beta * ((img[:, 1:] - img[:, :-1])**2).sum(axis=-1))
        upleftW[1:, 1:] = self.gamma/np.sqrt(2) * np.exp(-self.beta * ((img[1:, 1:] - img[:-1, :-1])**2).sum(axis=-1))
        upW[1:, :] = self.gamma * np.exp(-self.beta * ((img[1:, :] - img[:-1, :])**2).sum(axis=-1))
        uprightW[1:, :-1] = self.gamma/np.sqrt(2) * np.exp(-self.beta * ((img[1:, :-1] - img[:-1, 1:])**2).sum(axis=-1))

        return leftW, upleftW, upW, uprightW


    def calculateBeta(self, img):
        '''
        Calculates beta as per the paper.
        '''

        img = img.astype(int)

        _beta = 0
        
        # left
        _beta += ((img[:, 1:] - img[:, :-1])**2).sum()

        #upleft
        _beta += ((img[1:, 1:] - img[:-1, :-1])**2).sum()

        # u[]
        _beta += ((img[1:, :] - img[:-1, :])**2).sum()

        # upright
        _beta += ((img[1:, :-1] - img[:-1, 1:])**2).sum() 

        if _beta <= 1e-10:
            self.beta = 0
        else:
            self.beta = 1./(2 * _beta/(4 * img.shape[0] * img.shape[1] - 3 * img.shape[0] - 3 * img.shape[1] + 2))
        
    
    def constructGCGraph(self, img, mask, bgdGMM, fgdGMM, Lambda, leftW, upleftW, upW, uprightW, iseightconn=True):
        '''
        Constructs the graph for Grabcut using the Unary and Pairwise Potentials. Returns a Graph with the capacities.
        '''
        
        img = img.astype(int)

        rows, cols = mask.shape

        G = Graph()
        G.add_vertices(rows * cols + 2)

        capacities = []
        edges = []

        fromSources = -bgdGMM.score_samples(img.reshape(-1, 3))
        toSinks = -fgdGMM.score_samples(img.reshape(-1, 3))

        for i in range(rows):
            for j in range(cols):

                vtxId = i * cols + j + 1

                if mask[i, j] == self.PR_BGD or mask[i, j] == self.PR_FGD:
                    from_source = fromSources[vtxId]
                    to_sink = toSinks[vtxId]
                elif mask[i, j] == self.BGD:
                    from_source = 0
                    to_sink = Lambda
                else:
                    from_source = Lambda
                    to_sink = 0
                
                edges.append((0, vtxId))
                capacities.append(from_source)
                edges.append((vtxId, rows * cols + 1))
                capacities.append(to_sink)

                if j > 0:
                    edges.append((vtxId, vtxId - 1))
                    capacities.append(leftW[i, j])
                
                if j > 0 and i > 0:
                    edges.append((vtxId, vtxId - cols - 1))
                    capacities.append(upleftW[i, j])

                if i > 0 and iseightconn:
                    edges.append((vtxId, vtxId - cols))
                    capacities.append(upW[i, j])

                if j < cols - 1 and i > 0 and iseightconn:
                    edges.append((vtxId, vtxId - cols + 1))
                    capacities.append(uprightW[i, j])

        G.add_edges(edges)
        G.es['capacity'] = capacities
        
        return G

    def assign_and_learn_GMM(self, img, mask, n_components=5):
        '''
        Performs Step 1 and Step 2 of the Algorithm.
        Assigns points and learns GMM Params.
        '''

        bgdGMM = GaussianMixture(n_components=n_components)
        bgdGMM.fit(img[np.where((mask == self.BGD) | (mask == self.PR_BGD))])

        fgdGMM = GaussianMixture(n_components=n_components)
        fgdGMM.fit(img[np.where((mask == self.FGD) | (mask == self.PR_FGD))])

        return bgdGMM, fgdGMM

    def estimateSegmentation(self, graph, mask):
        '''
        Performs Step 3 of the algorithm.
        Performs Mincut algorithm and reassigns the mask.
        '''
        c = graph.mincut(source=0, target=mask.shape[0] * mask.shape[1] + 1, capacity=graph.es['capacity'])

        fg = np.array(c[0][1:]) - 1
        bg = np.array(c[1][:-1]) - 1
        
        mask2 = mask.copy().flatten()
        mask2[fg] = 1
        mask2[bg] = 0
        mask2 = mask2.reshape(mask.shape)

        rows, cols = mask.shape

        mask[np.where(((mask == self.PR_BGD) | (mask == self.PR_FGD)) & (mask2 == 1))] = self.PR_FGD
        mask[np.where(((mask == self.PR_BGD) | (mask == self.PR_FGD)) & (mask2 == 0))] = self.PR_BGD

        return mask

    def obtainFinalMask(self, mask):
        '''
        Obtains the final mask for output with just two classes FGD and BGD.
        '''
        mask2 = mask.copy()
        mask2[np.where(mask2 == 3)] = 1
        mask2[np.where(mask2 == 2)] = 0
        
        return mask2

    
    def calcEs(self, img, mask, bgdGMM, fgdGMM, leftW, upleftW, upW, uprightW):
        '''
        Calculates the Energy given all the required params.
        '''
        mask2 = self.obtainFinalMask(mask)

        bgd_U = -bgdGMM.score_samples(img[np.where(mask2 == 0)]).sum()
        fgd_U = -fgdGMM.score_samples(img[np.where(mask2 == 1)]).sum()

        total_U = bgd_U + fgd_U

        leftV = leftW[:, 1:] * (mask2[:, 1:] != mask2[:, :-1])
        upleftV = upleftW[1:, 1:] * (mask2[1:, 1:] != mask2[:-1, :-1])
        upV = upW[1:, :] * (mask2[1:, :] != mask2[:-1, :])
        uprightV = uprightW[1:, :-1] * (mask2[1:, :-1]  != mask2[:-1, 1:])

        total_V = leftV.sum() + upleftV.sum() + upV.sum() + uprightV.sum()

        return total_U + total_V

def plot_imgpair(img1, img2):
    '''
    Plots Image Pairs.
    '''
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(20, 10 * img1.shape[1]/img1.shape[0]))
    
    plt.subplot(121)
    plt.imshow(img1)
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(img2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()