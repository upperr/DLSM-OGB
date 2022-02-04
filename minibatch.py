from __future__ import division
from __future__ import print_function
import numpy as np

class EdgeMinibatchIterator(object):
    
    """ This minibatch iterator iterates over batches of sampled edges.

    train_edges -- array of training edges
    val_edges -- array of validation edges
    test_edges -- array of testing edges
    placeholders -- tensorflow placeholders object
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, train_edges, val_edges, test_edges, placeholders, 
                 batch_size = 100, **kwargs):

        self.train_edges = train_edges['pos']
        self.train_neg_edges = train_edges['neg']
        if isinstance(val_edges, dict):
            self.val_edges = val_edges['pos']
            self.val_neg_edges = val_edges['neg']
            self.test_edges = test_edges['pos']
            self.test_neg_edges = test_edges['neg']
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.batch_num = 0

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_pos_edges, batch_neg_edges):
        nodes_source = np.hstack([batch_pos_edges[:, 0], batch_neg_edges[:, 0]])
        nodes_target = np.hstack([batch_pos_edges[:, 1], batch_neg_edges[:, 1]])
        labels = np.hstack([np.ones([len(batch_pos_edges)], np.float32), 
                            np.zeros([len(batch_neg_edges)], np.float32)])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_pos_edges) + len(batch_neg_edges)})
        feed_dict.update({self.placeholders['batch_nodes_source']: nodes_source})
        feed_dict.update({self.placeholders['batch_nodes_target']: nodes_target})
        feed_dict.update({self.placeholders['labels'] : labels})

        return feed_dict, labels

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        batch_neg_edges = self.train_neg_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges, batch_neg_edges)

    def val_feed_dict(self, size = None):
        '''Randomly select a batch of edges from the validate set'''
        edge_list = self.val_edges
        false_edge_list = self.val_neg_edges
            
        if size is None:
            feed_dict, labels = self.batch_feed_dict(edge_list, false_edge_list)
        else:
            size_false = int(size * len(false_edge_list) / (len(edge_list) + len(false_edge_list)))
            ind = np.random.permutation(len(edge_list))
            batch_edges = edge_list[ind[: min(size - size_false, len(ind))], :]
            ind = np.random.permutation(len(false_edge_list))
            batch_false_edges = false_edge_list[ind[: min(size_false, len(ind))], :]
            feed_dict, labels = self.batch_feed_dict(batch_edges, batch_false_edges)
            
        return feed_dict, labels

    def incremental_test_feed_dict(self, size, iter_num):
        '''Incrementally select a batch of edges from the testing set'''
        edge_list = self.test_edges
        false_edge_list = self.test_neg_edges
        size_true = int(size * len(edge_list) / (len(edge_list) + len(false_edge_list)))
        size_false = size - size_true
        batch_edges = edge_list[iter_num * size_true: min((iter_num + 1) * size_true, len(edge_list))]
        batch_false_edges = false_edge_list[iter_num * size_false: min((iter_num + 1) * size_false, len(false_edge_list))]
        feed_dict, labels = self.batch_feed_dict(batch_edges, batch_false_edges)
        finished = (iter_num + 1) * size >= (len(edge_list) + len(false_edge_list))
        
        return feed_dict, labels, finished

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        #self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0
