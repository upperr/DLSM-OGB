import numpy as np
from ogb.linkproppred import LinkPropPredDataset
from ogb.nodeproppred import NodePropPredDataset
from tqdm import tqdm

def load_data(dataset, max_degree):
     
    if dataset.startswith('ogbl'):
        train_edges, val_edges, test_edges, features, neighbors_train, neighbors = load_edge_data(dataset, max_degree)
        return train_edges, val_edges, test_edges, None, features, neighbors_train, neighbors
    elif dataset.startswith('ogbn'):
        train_edges, labels, features, neighbors_train = load_node_data(dataset, max_degree)
        return train_edges, None, None, labels, features, neighbors_train, None
    else:
        raise NameError('No data named ' + dataset + '!')

def load_edge_data(dataset, max_degree):
    
    dataset = LinkPropPredDataset(name = dataset)
    split_edge = dataset.get_edge_split()
    if 'num_nodes' in dataset[0].keys():
        num_nodes = dataset[0]['num_nodes']
        features = dataset[0]['node_feat']
    elif 'num_nodes_dict' in dataset[0].keys():
        num_nodes = sum(list(dataset[0]['num_nodes_dict'].values()))
        features = dataset[0]['node_feat_dict']
    del dataset
    
    if 'source_node' in split_edge['train'].keys():
        # positive edges
        train_pos_edges = np.vstack([split_edge['train']['source_node'], split_edge['train']['target_node']]).T
        val_pos_edges = np.vstack([split_edge['valid']['source_node'], split_edge['valid']['target_node']]).T
        test_pos_edges = np.vstack([split_edge['test']['source_node'], split_edge['test']['target_node']]).T
        # negative edges: use one negative edge for each positive edge
        val_neg_edges = np.vstack([split_edge['valid']['source_node'], split_edge['valid']['target_node_neg'][:, 0]]).T
        test_neg_edges = np.vstack([split_edge['test']['source_node'], split_edge['test']['target_node_neg'][:, 0]]).T
    elif 'head' in split_edge['train'].keys():
        # positive edges
        train_pos_edges = np.vstack([split_edge['train']['head'], split_edge['train']['tail']]).T
        val_pos_edges = np.vstack([split_edge['valid']['head'], split_edge['valid']['tail']]).T
        test_pos_edges = np.vstack([split_edge['test']['head'], split_edge['test']['tail']]).T
        # negative edges: use one negative edge for each positive edge
        val_neg_edges = np.vstack([split_edge['valid']['head'], split_edge['valid']['tail_neg'][:, 0]]).T
        test_neg_edges = np.vstack([split_edge['test']['head'], split_edge['test']['tail_neg'][:, 0]]).T
    del split_edge
    
    print('Sampling node neighbors..')
    neighbors_train = {'out': {node: [] for node in range(num_nodes)},
                       'in': {node: [] for node in range(num_nodes)}}
    for s, t in train_pos_edges:
        neighbors_train['out'][s].append(t)
        neighbors_train['in'][t].append(s)
    neighbors = neighbors_train
    for s, t in val_pos_edges:
        neighbors['out'][s].append(t)
        neighbors['in'][t].append(s)
    for s, t in test_pos_edges:
        neighbors['out'][s].append(t)
        neighbors['in'][t].append(s)
    neighbors_train['out'], neighbors_train['in'] = sample_neighbors(neighbors_train, max_degree)
    neighbors['out'], neighbors['in'] = sample_neighbors(neighbors, max_degree)

    print('Sampling negative edges for training..')
    train_neg_target = np.random.randint(0, num_nodes, size = len(train_pos_edges))
    for i in tqdm(range(len(train_pos_edges)), ncols = 70):
        node_neighbors = np.hstack([neighbors['out'][train_pos_edges[i, 0]], neighbors['in'][train_pos_edges[i, 0]]])
        while train_neg_target[i] in node_neighbors:
            train_neg_target[i] = np.random.randint(0, num_nodes, size = 1)
    train_neg_edges = np.vstack([train_pos_edges[:, 0], train_neg_target]).T
    
    train_edges = {'pos': train_pos_edges, 'neg': train_neg_edges}
    val_edges = {'pos': val_pos_edges, 'neg': val_neg_edges}
    test_edges = {'pos': test_pos_edges, 'neg': test_neg_edges}
    
    return train_edges, val_edges, test_edges, features, neighbors_train, neighbors

def load_node_data(dataset, max_degree):
    
    dataset = NodePropPredDataset(name = dataset)
    graph, labels = dataset[0] # graph: library-agnostic graph object
    if labels.shape[1] > 1:
        raise Warning('Transform binary labels to multi-class labels!')
        labels = [np.random.choice(np.reshape(np.argwhere(i), [-1])) for i in labels]
    del dataset
    
    if 'num_nodes' in graph.keys():
        num_nodes = graph['num_nodes']
        features = graph['node_feat']
    elif 'num_nodes_dict' in graph.keys():
        num_nodes = sum(list(graph['num_nodes_dict'].values()))
        features = graph['node_feat_dict']
    
    edges_pos = graph['edge_index'].T
    
    print('Sampling node neighbors..')
    neighbors = {'out': {node: [] for node in range(num_nodes)}, 
                 'in': {node: [] for node in range(num_nodes)}}
    for s, t in edges_pos:
        neighbors['out'][s].append(t)
        neighbors['in'][t].append(s)
    neighbors['out'], neighbors['in'] = sample_neighbors(neighbors, max_degree)
    
    print('Sampling negative edges for training..')
    edges_neg_target = np.random.randint(0, num_nodes, size = len(edges_pos))
    for i in tqdm(range(len(edges_pos)), ncols = 70):
        node_neighbors = np.hstack([neighbors['out'][edges_pos[i, 0]], neighbors['in'][edges_pos[i, 0]]])
        while edges_neg_target[i] in node_neighbors:
            edges_neg_target[i] = np.random.randint(0, num_nodes, size = 1)
    edges_neg = np.vstack([edges_pos[:, 0], edges_neg_target]).T
    
    return {'pos': edges_pos, 'neg': edges_neg}, np.reshape(labels, [-1]), features, neighbors

def sample_neighbors(neighbors, max_degree):
    nodes = list(range(len(neighbors['out'])))
    neighbors_out_sample = len(nodes) * np.ones((len(nodes) + 1, max_degree), dtype = np.int16)
    neighbors_in_sample = len(nodes) * np.ones((len(nodes) + 1, max_degree), dtype = np.int16)
        
    for nodeid in tqdm(nodes, ncols = 70):
        # sample out-neighbors
        neighbors_out = neighbors['out'][nodeid]
        if len(neighbors_out) == 0:
            continue
        if len(neighbors_out) > max_degree:
            neighbors_out_sample[nodeid, :] = np.random.choice(neighbors_out, max_degree, replace = False)
        elif len(neighbors_out) < max_degree:
            neighbors_out_sample[nodeid, :] = np.random.choice(neighbors_out, max_degree, replace = True)
        # sample in-neighbors        
        neighbors_in = neighbors['in'][nodeid]
        if len(neighbors_in) == 0:
            continue
        if len(neighbors_in) > max_degree:
            neighbors_in_sample[nodeid, :] = np.random.choice(neighbors_in, max_degree, replace = False)
        elif len(neighbors_out) < max_degree:
            neighbors_in_sample[nodeid, :] = np.random.choice(neighbors_in, max_degree, replace = True)
                
    return neighbors_out_sample, neighbors_in_sample