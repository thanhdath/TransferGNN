from __future__ import print_function
import numpy as np
import multiprocessing
import networkx as nx


def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)

def deepwalk_walk(args):
    '''
    Simulate a random walk starting from start node.
    '''
    walk_length = args["walk_length"]
    neibs = args["neibs"]
    nodes = args["nodes"]
    if args["iter"] % 100 == 0:
        print("Iter:", args["iter"])

    walks = []
    for node in nodes:
        walk = [str(node)]
        if len(neibs[node]) == 0:
            walks.append(walk)
            continue
        while len(walk) < walk_length:
            cur = int(walk[-1])
            cur_nbrs = neibs[cur]
            if len(cur_nbrs) == 0: break
            walk.append(str(np.random.choice(cur_nbrs)))
        walks.append(walk)
    return walks


class BasicWalker:
    def __init__(self, neibs_dict, workers):
        self.neibs = neibs_dict
        self.nodes = list(set(self.neibs.keys()))

    def simulate_walks(self, num_walks, walk_length, num_workers):
        pool = multiprocessing.Pool(processes=num_workers)
        walks = []
        print('Walk iteration:')
        nodes = self.nodes
        nodess = [np.random.shuffle(nodes)]
        for i in range(num_walks):
            _ns = nodes.copy()
            np.random.shuffle(_ns)
            nodess.append(_ns)
        params = list(map(lambda x: {'walk_length': walk_length, 'neibs': self.neibs, 'iter': x, 'nodes': nodess[x]},
            list(range(1, num_walks+1))))
        walks = pool.map(deepwalk_walk, params)
        pool.close()
        pool.join()
        # walks = np.vstack(walks)
        while len(walks) > 1:
            walks[-2] = walks[-2] + walks[-1]
            walks = walks[:-1]
        walks = walks[0]

        return walks


class DegreeCorrelationWalker:
    def __init__(self, neibs_dict, workers):
        self.neibs = neibs_dict
        self.nodes = list(set(self.neibs.keys()))
        self.degrees_dict = {k: len(v) for k, v in self.neibs.items()}
        self.average_degree = np.mean(list(self.degrees_dict.values()))

    def simulate_walks(self, average_walks, min_walks, walk_length, num_workers):
        pool = multiprocessing.Pool(processes=num_workers)
        walks = []
        print('Walk iteration:')
        # nodess = [np.random.shuffle(nodes)]
        # for i in range(num_walks):
        #     _ns = nodes.copy()
        #     np.random.shuffle(_ns)
        #     nodess.append(_ns)
        b = min_walks
        a = (average_walks - b) / self.average_degree
        nodess = []
        for node in self.nodes:
            n_walk = a * self.degrees_dict[node] + b
            nodess += [node] * int(n_walk)
        nodess = np.array(nodess).tolist()
        # TODO: FIX here
        bs = 100
        params = list(map(lambda x: {
            'walk_length': walk_length, 
            'neibs': self.neibs, 
            'iter': x, 
            'nodes': nodess[bs*x:bs*(x+1)]},
            list(range(0, len(nodess)//bs + 1))))

        walks = pool.map(deepwalk_walk, params)
        pool.close()
        pool.join()
        # walks = np.vstack(walks)
        while len(walks) > 1:
            walks[-2] = walks[-2] + walks[-1]
            walks = walks[:-1]
        walks = walks[0]

        return walks


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
