from layers import FullConnection, GraphConvolution, LSMDecoder
import tensorflow as tf
from utils import *

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
SMALL = 1e-16

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass
   
class DLSM(Model):
    def __init__(self, placeholders, neighbors_out, neighbors_in, features, 
                 mc_samples = 1, identity_dim = 0, **kwargs):
        super().__init__(**kwargs)

        self.num_neighbors = [int(x) for x in FLAGS.num_neighbors.split('_')]
        self.neighbors_out = neighbors_out
        self.neighbors_in = neighbors_in
        
        if identity_dim > 0:
           self.embeds = tf.compat.v1.get_variable("node_embeddings", [neighbors_out.get_shape().as_list()[0] + 1, identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.inputs = self.embeds
            self.input_dim = identity_dim
        else:
            self.inputs = tf.Variable(tf.constant(features, dtype = tf.float32), trainable = False)
            self.input_dim = features.shape[1]
            if not self.embeds is None:
                self.inputs = tf.concat([self.embeds, self.inputs], axis = 1)
                self.input_dim += identity_dim
        
        self.nodes1 = placeholders['batch_nodes_source']
        self.nodes2 = placeholders['batch_nodes_target']
        self.batch_size = placeholders['batch_size']
        self.dropout = placeholders['dropout']

        self.encoder_layers = [int(x) for x in FLAGS.encoder.split('_')]
        self.decoder_layers = [int(x) for x in FLAGS.decoder.split('_')]
        self.num_encoder_layers = len(self.encoder_layers)
        self.num_decoder_layers = len(self.decoder_layers)
        
        self.prior_theta_param = []
        self.posterior_theta_param = []
        self.S = mc_samples #No. of MC samples
        
        self.build()
    
    def uniform_neighbor_sampler(self, inputs):
        """
        Uniformly samples neighbors.
        Assumes that adj lists are padded with random re-sampling
        """
        ids, num_samples = inputs
        
        neighbors_out = tf.nn.embedding_lookup(self.neighbors_out, ids)
        neighbors_out = tf.transpose(tf.random.shuffle(tf.transpose(neighbors_out)))
        neighbors_out_samples = tf.slice(neighbors_out, [0, 0], [-1, num_samples])
        
        neighbors_in = tf.nn.embedding_lookup(self.neighbors_in, ids)
        neighbors_in = tf.transpose(tf.random.shuffle(tf.transpose(neighbors_in)))
        neighbors_in_samples = tf.slice(neighbors_in, [0, 0], [-1, num_samples])
        
        return neighbors_out_samples, neighbors_in_samples
    
    def sample(self, nodes, batch_size = None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.
        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        if batch_size is None:
            batch_size = self.batch_size
        samples = [nodes]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(self.num_encoder_layers):
            t = self.num_encoder_layers - k - 1
            support_size *= 2 * self.num_neighbors[t]
            support_sizes.append(support_size)
            neighbors_out, neighbors_in = self.uniform_neighbor_sampler((samples[k], self.num_neighbors[t]))
            neighbors = tf.reshape(tf.concat([neighbors_out, neighbors_in], 1), [support_size * batch_size, ])
            samples.append(neighbors)

        return samples, support_sizes
    
    def aggregate(self, inputs, support_sizes, layer, gc, batch_size = None):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            inputs: the input features for each sample of various hops away.
            sample_indices: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            support_sizes: the number of nodes to gather information from for each layer.
            layer: current layer
            gc: the graph convolution network to aggregate hidden representations
        Returns:
            a list of hidden representations for each sample of various hops away at the current layer
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        h_hop = []
        for hop in range(self.num_encoder_layers - layer):
            nei = tf.reshape(inputs[hop + 1], [batch_size * support_sizes[hop], # batch size
                                               2 * self.num_neighbors[self.num_encoder_layers - hop - 1], # number of neighbors
                                               -1]) # dimension of input hidden representations
            nei_out = tf.slice(nei, 
                               begin = [0, 0, 0], 
                               size = [-1, self.num_neighbors[self.num_encoder_layers - hop - 1], -1])
            nei_in = tf.slice(nei,
                              begin = [0, self.num_neighbors[self.num_encoder_layers - hop - 1], 0], 
                              size = [-1, self.num_neighbors[self.num_encoder_layers - hop - 1], -1])
            h = gc((inputs[hop], # self
                    nei_out, # out-neighbors
                    nei_in)) # in-neighbors
            h_hop.append(h)
        
        return h_hop
    
    def _build(self):

        print('Build Dynamic Network....')
        
        samples1, support_sizes1 = self.sample(self.nodes1)
        samples2, support_sizes2 = self.sample(self.nodes2)
        inputs1 = [tf.nn.embedding_lookup(self.inputs, node_samples) for node_samples in samples1]
        inputs2 = [tf.nn.embedding_lookup(self.inputs, node_samples) for node_samples in samples2]

        self.posterior_theta_param1 = []
        self.posterior_theta_param2 = []
        self.h1 = []
        self.h2 = []
        self.layers = []

        # GraphSAGE Encoder
        for idx, encoder_layer in enumerate(self.encoder_layers):

            act = tf.nn.sigmoid
            if idx == 0:
                gc_input = GraphConvolution(input_dim = self.input_dim,
                                            output_dim = encoder_layer,
                                            act = act,
                                            dropout = self.dropout,
                                            name = 'conv_input_' + str(idx),
                                            logging = self.logging)
                h1 = self.aggregate(inputs = inputs1,
                                   support_sizes = support_sizes1, 
                                   layer = idx, 
                                   gc = gc_input)
                h2 = self.aggregate(inputs = inputs2,
                                   support_sizes = support_sizes2, 
                                   layer = idx, 
                                   gc = gc_input)
                
                self.h1.append([h1, h1, h1, h1])
                self.h2.append([h2, h2, h2, h2])

            else:
                gc_mean = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                           output_dim = encoder_layer,
                                           act = act,
                                           dropout = self.dropout,
                                           name = 'conv_mean_' + str(idx),
                                           logging = self.logging)
                h_mean1 = self.aggregate(inputs = self.h1[-1][0],
                                        support_sizes = support_sizes1, 
                                        layer = idx, 
                                        gc = gc_mean)
                h_mean2 = self.aggregate(inputs = self.h2[-1][0],
                                        support_sizes = support_sizes2, 
                                        layer = idx, 
                                        gc = gc_mean)
                
                gc_std = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                           output_dim = encoder_layer,
                                           act = act,
                                           dropout = self.dropout,
                                           name = 'conv_std_' + str(idx),
                                           logging = self.logging)
                h_std1 = self.aggregate(inputs = self.h1[-1][1],
                                       support_sizes = support_sizes1, 
                                       layer = idx, 
                                       gc = gc_std)
                h_std2 = self.aggregate(inputs = self.h2[-1][1],
                                       support_sizes = support_sizes2, 
                                       layer = idx, 
                                       gc = gc_std)
                
                gc_pi = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                           output_dim = encoder_layer,
                                           act = act,
                                           dropout = self.dropout,
                                           name = 'conv_pi_' + str(idx),
                                           logging = self.logging)
                h_pi1 = self.aggregate(inputs = self.h1[-1][2],
                                       support_sizes = support_sizes1, 
                                       layer = idx, 
                                       gc = gc_pi)
                h_pi2 = self.aggregate(inputs = self.h2[-1][2],
                                       support_sizes = support_sizes2, 
                                       layer = idx, 
                                       gc = gc_pi)
                
                gc_alpha_gam = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                           output_dim = encoder_layer,
                                           act = act,
                                           dropout = self.dropout,
                                           name = 'conv_alpha_gam_' + str(idx),
                                           logging = self.logging)
                h_alpha_gam = self.aggregate(inputs = self.h1[-1][3],
                                       support_sizes = support_sizes1, 
                                       layer = idx, 
                                       gc = gc_alpha_gam)
                
                if FLAGS.directed == 1:
                    gc_alpha_del = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                           output_dim = encoder_layer,
                                           act = act,
                                           dropout = self.dropout,
                                           name = 'conv_alpha_del_' + str(idx),
                                           logging = self.logging)
                    h_alpha_del = self.aggregate(inputs = self.h2[-1][3],
                                       support_sizes = support_sizes2, 
                                       layer = idx, 
                                       gc = gc_alpha_del)
                    
                else:
                    h_alpha_del = h_alpha_gam
                    
                self.h1.append([h_mean1, h_std1, h_pi1, h_alpha_gam])
                self.h2.append([h_mean2, h_std2, h_pi2, h_alpha_del])

                # get Theta parameters
                mean_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                z_mean1 = mean_layer(self.h1[-1][0][0])
                z_mean2 = mean_layer(self.h2[-1][0][0])
                
                std_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                z_std1 = std_layer(self.h1[-1][1][0])
                z_std2 = std_layer(self.h2[-1][1][0])
                
                pi_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                pi_logit1 = pi_layer(self.h1[-1][2][0])
                pi_logit2 = pi_layer(self.h2[-1][2][0])
                
                alpha_gam_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: tf.nn.softplus(x))
                alpha_gam = alpha_gam_layer(self.h1[-1][3][0])
            
                if FLAGS.directed == 1:
                    alpha_del_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: tf.nn.softplus(x))
                    alpha_del = alpha_del_layer(self.h2[-1][3][0])
                else:
                    alpha_del = alpha_gam
                    
                self.posterior_theta_param1.append([z_mean1, z_std1, pi_logit1, alpha_gam])
                self.posterior_theta_param2.append([z_mean2, z_std2, pi_logit2, alpha_del])

        ###########################################################################
        # HLSM Decoder
        self.theta1_list = []
        self.theta2_list = []
        self.reconstructions_list = []
        self.posterior_theta_param1_list = []
        self.posterior_theta_param2_list = []
        self.prior_theta_param_list = []
        #Take multiple MC samples
        for k in range(self.S):
            self.theta1 = []
            self.theta2 = []
            self.prior_theta_param = []
            # Downward Inference Pass
            for idx, decoder_layer in enumerate(self.decoder_layers): # l = 1, 2, ..., L-1
                
                if idx == 0:
                    mean_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param1[idx][0] += mean_layer(self.h1[-1][0][0])
                    self.posterior_theta_param2[idx][0] += mean_layer(self.h2[-1][0][0])
                    
                    pi_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param1[idx][2] += pi_layer(self.h1[-1][2][0])
                    self.posterior_theta_param2[idx][2] += pi_layer(self.h2[-1][2][0])
                    
                    alpha_gam_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                    self.posterior_theta_param1[idx][3] += alpha_gam_layer(self.h1[-1][3][0])
                    
                    if FLAGS.directed == 1:
                        alpha_del_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                        self.posterior_theta_param2[idx][3] += alpha_del_layer(self.h2[-1][3][0])
                    else:
                        self.posterior_theta_param2[idx][3] = self.posterior_theta_param1[idx][3]
                
                else:
                    mean_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param1[idx][0] += mean_layer(self.theta1[idx - 1][0])
                    self.posterior_theta_param2[idx][0] += mean_layer(self.theta2[idx - 1][0])
                    
                    pi_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param1[idx][2] += pi_layer(self.theta1[idx - 1][1])
                    self.posterior_theta_param2[idx][2] += pi_layer(self.theta2[idx - 1][1])
                    
                    alpha_gam_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                    self.posterior_theta_param1[idx][3] += alpha_gam_layer(self.theta1[idx - 1][3])
                    
                    if FLAGS.directed == 1:
                        alpha_del_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                        self.posterior_theta_param2[idx][3] += alpha_del_layer(self.theta2[idx - 1][3])
                    else:
                        self.posterior_theta_param2[idx][3] = self.posterior_theta_param1[idx][3]
                # community membership
                v = tf.constant(FLAGS.v0, shape = (1, decoder_layer))
                pi_logit_prior = logit(tf.exp(tf.cumsum(tf.math.log(v + SMALL), axis = 1)))
                s_logit1 = sample_binconcrete(self.posterior_theta_param1[idx][2], FLAGS.temp_post)
                s1 = tf.nn.sigmoid(s_logit1)
                s_logit2 = sample_binconcrete(self.posterior_theta_param2[idx][2], FLAGS.temp_post)
                s2 = tf.nn.sigmoid(s_logit2)
                # latent position
                z1 = sample_normal(self.posterior_theta_param1[idx][0], self.posterior_theta_param1[idx][1])
                z1 = tf.multiply(s1, z1)
                z2 = sample_normal(self.posterior_theta_param2[idx][0], self.posterior_theta_param2[idx][1])
                z2 = tf.multiply(s2, z2)
                # node random factors
                alpha_gam_prior = tf.constant(1. / decoder_layer, tf.float32)
                # composing Gamma distribution (with same beta) as Dirichlet distribution
                gamma = sample_gamma(self.posterior_theta_param1[idx][3], tf.constant(FLAGS.beta, tf.float32)) 
                gamma = gamma / (tf.reduce_sum(gamma, axis = 0) + SMALL)
                gamma = tf.multiply(s1, gamma)
            
                if FLAGS.directed == 1:
                    alpha_del_prior = tf.constant(1. / decoder_layer, tf.float32)
                    # composing Gamma distribution (with same beta) as Dirichlet distribution
                    delta = sample_gamma(self.posterior_theta_param2[idx][3], tf.constant(FLAGS.beta, tf.float32))
                    delta = delta / (tf.reduce_sum(delta, axis = 0) + SMALL)
                    delta = tf.multiply(s2, delta)
                else:
                    delta = gamma
                
                self.prior_theta_param.append([pi_logit_prior, alpha_gam_prior, alpha_del_prior])
                self.theta1.append([z1, s_logit1, s1, gamma])
                self.theta2.append([z2, s_logit2, s2, delta])

            output_layer = LSMDecoder(input_dim = self.decoder_layers[-1], num_edges = self.batch_size, act = lambda x: tf.nn.sigmoid(x), logging = self.logging)
            self.reconstructions, self.theta_decoder = output_layer([[self.theta1[-1][0], self.theta1[-1][3]], 
                                                                     [self.theta2[-1][0], self.theta2[-1][3]]])
            
            self.theta1_list.append(self.theta1)
            self.theta2_list.append(self.theta2)
            self.reconstructions_list.append(self.reconstructions)
            self.posterior_theta_param1_list.append(self.posterior_theta_param1)
            self.posterior_theta_param2_list.append(self.posterior_theta_param2)
            self.prior_theta_param_list.append(self.prior_theta_param)
            
        ###############################################################

class DLSM_D(Model):
    def __init__(self, placeholders, neighbors_out, neighbors_in, features, 
                 mc_samples = 1, identity_dim = 0, **kwargs):
        super().__init__(**kwargs)

        self.num_neighbors = [int(x) for x in FLAGS.num_neighbors.split('_')]
        self.neighbors_out = neighbors_out
        self.neighbors_in = neighbors_in
        
        if identity_dim > 0:
           self.embeds = tf.compat.v1.get_variable("node_embeddings", [neighbors_out.get_shape().as_list()[0] + 1, identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.inputs = self.embeds
            self.input_dim = identity_dim
        else:
            self.inputs = tf.Variable(tf.constant(features, dtype = tf.float32), trainable = False)
            self.input_dim = features.shape[1]
            if not self.embeds is None:
                self.inputs = tf.concat([self.embeds, self.inputs], axis = 1)
                self.input_dim += identity_dim
        
        self.nodes1 = placeholders['batch_nodes_source']
        self.nodes2 = placeholders['batch_nodes_target']
        self.batch_size = placeholders['batch_size']
        self.dropout = placeholders['dropout']

        self.encoder_layers = [int(x) for x in FLAGS.encoder.split('_')]
        self.decoder_layers = [int(x) for x in FLAGS.decoder.split('_')]
        self.num_encoder_layers = len(self.encoder_layers)
        self.num_decoder_layers = len(self.decoder_layers)
        
        self.prior_theta_param = []
        self.posterior_theta_param = []
        self.S = mc_samples #No. of MC samples
        
        self.build()
    
    def uniform_neighbor_sampler(self, inputs):
        """
        Uniformly samples neighbors.
        Assumes that adj lists are padded with random re-sampling
        """
        ids, num_samples = inputs
        
        neighbors_out = tf.nn.embedding_lookup(self.neighbors_out, ids)
        neighbors_out = tf.transpose(tf.random.shuffle(tf.transpose(neighbors_out)))
        neighbors_out_samples = tf.slice(neighbors_out, [0, 0], [-1, num_samples])
        
        neighbors_in = tf.nn.embedding_lookup(self.neighbors_in, ids)
        neighbors_in = tf.transpose(tf.random.shuffle(tf.transpose(neighbors_in)))
        neighbors_in_samples = tf.slice(neighbors_in, [0, 0], [-1, num_samples])
        
        return neighbors_out_samples, neighbors_in_samples
    
    def sample(self, nodes, batch_size = None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.
        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        if batch_size is None:
            batch_size = self.batch_size
        samples = [nodes]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(self.num_encoder_layers):
            t = self.num_encoder_layers - k - 1
            support_size *= 2 * self.num_neighbors[t]
            support_sizes.append(support_size)
            neighbors_out, neighbors_in = self.uniform_neighbor_sampler((samples[k], self.num_neighbors[t]))
            neighbors = tf.reshape(tf.concat([neighbors_out, neighbors_in], 1), [support_size * batch_size, ])
            samples.append(neighbors)

        return samples, support_sizes
    
    def aggregate(self, inputs, support_sizes, layer, gc, batch_size = None):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            inputs: the input features for each sample of various hops away.
            sample_indices: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            support_sizes: the number of nodes to gather information from for each layer.
            layer: current layer
            gc: the graph convolution network to aggregate hidden representations
        Returns:
            a list of hidden representations for each sample of various hops away at the current layer
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        h_hop = []
        for hop in range(self.num_encoder_layers - layer):
            nei = tf.reshape(inputs[hop + 1], [batch_size * support_sizes[hop], # batch size
                                               2 * self.num_neighbors[self.num_encoder_layers - hop - 1], # number of neighbors
                                               -1]) # dimension of input hidden representations
            nei_out = tf.slice(nei, 
                               begin = [0, 0, 0], 
                               size = [-1, self.num_neighbors[self.num_encoder_layers - hop - 1], -1])
            nei_in = tf.slice(nei,
                              begin = [0, self.num_neighbors[self.num_encoder_layers - hop - 1], 0], 
                              size = [-1, self.num_neighbors[self.num_encoder_layers - hop - 1], -1])
            h = gc((inputs[hop], # self
                    nei_out, # out-neighbors
                    nei_in)) # in-neighbors
            h_hop.append(h)
        
        return h_hop
    
    def _build(self):

        print('Build Dynamic Network....')
        
        samples1, support_sizes1 = self.sample(self.nodes1)
        samples2, support_sizes2 = self.sample(self.nodes2)
        inputs1 = [tf.nn.embedding_lookup(self.inputs, node_samples) for node_samples in samples1]
        inputs2 = [tf.nn.embedding_lookup(self.inputs, node_samples) for node_samples in samples2]

        self.posterior_theta_param1 = []
        self.posterior_theta_param2 = []
        self.h1 = []
        self.h2 = []
        self.layers = []

        # GraphSAGE Encoder
        for idx, encoder_layer in enumerate(self.encoder_layers):

            act = tf.nn.sigmoid
            if idx == 0:
                gc_input = GraphConvolution(input_dim = self.input_dim,
                                            output_dim = encoder_layer,
                                            act = act,
                                            dropout = self.dropout,
                                            name = 'conv_input_' + str(idx),
                                            logging = self.logging)
                h1 = self.aggregate(inputs = inputs1,
                                   support_sizes = support_sizes1, 
                                   layer = idx, 
                                   gc = gc_input)
                h2 = self.aggregate(inputs = inputs2,
                                   support_sizes = support_sizes2, 
                                   layer = idx, 
                                   gc = gc_input)
                
                self.h1.append([h1, h1, h1])
                self.h2.append([h2, h2, h2])

            else:
                gc_mean = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                           output_dim = encoder_layer,
                                           act = act,
                                           dropout = self.dropout,
                                           name = 'conv_mean_' + str(idx),
                                           logging = self.logging)
                h_mean1 = self.aggregate(inputs = self.h1[-1][0],
                                        support_sizes = support_sizes1, 
                                        layer = idx, 
                                        gc = gc_mean)
                h_mean2 = self.aggregate(inputs = self.h2[-1][0],
                                        support_sizes = support_sizes2, 
                                        layer = idx, 
                                        gc = gc_mean)
                
                gc_std = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                           output_dim = encoder_layer,
                                           act = act,
                                           dropout = self.dropout,
                                           name = 'conv_std_' + str(idx),
                                           logging = self.logging)
                h_std1 = self.aggregate(inputs = self.h1[-1][1],
                                       support_sizes = support_sizes1, 
                                       layer = idx, 
                                       gc = gc_std)
                h_std2 = self.aggregate(inputs = self.h2[-1][1],
                                       support_sizes = support_sizes2, 
                                       layer = idx, 
                                       gc = gc_std)
                
                gc_alpha_gam = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                           output_dim = encoder_layer,
                                           act = act,
                                           dropout = self.dropout,
                                           name = 'conv_alpha_gam_' + str(idx),
                                           logging = self.logging)
                h_alpha_gam = self.aggregate(inputs = self.h1[-1][2],
                                       support_sizes = support_sizes1, 
                                       layer = idx, 
                                       gc = gc_alpha_gam)
                
                if FLAGS.directed == 1:
                    gc_alpha_del = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                           output_dim = encoder_layer,
                                           act = act,
                                           dropout = self.dropout,
                                           name = 'conv_alpha_del_' + str(idx),
                                           logging = self.logging)
                    h_alpha_del = self.aggregate(inputs = self.h2[-1][2],
                                       support_sizes = support_sizes2, 
                                       layer = idx, 
                                       gc = gc_alpha_del)
                    
                else:
                    h_alpha_del = h_alpha_gam
                    
                self.h1.append([h_mean1, h_std1, h_alpha_gam])
                self.h2.append([h_mean2, h_std2, h_alpha_del])

                # get Theta parameters
                mean_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                z_mean1 = mean_layer(self.h1[-1][0][0])
                z_mean2 = mean_layer(self.h2[-1][0][0])
                
                std_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                z_std1 = std_layer(self.h1[-1][1][0])
                z_std2 = std_layer(self.h2[-1][1][0])
                
                alpha_gam_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: tf.nn.softplus(x))
                alpha_gam = alpha_gam_layer(self.h1[-1][2][0])
            
                if FLAGS.directed == 1:
                    alpha_del_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: tf.nn.softplus(x))
                    alpha_del = alpha_del_layer(self.h2[-1][2][0])
                else:
                    alpha_del = alpha_gam
                    
                self.posterior_theta_param1.append([z_mean1, z_std1, alpha_gam])
                self.posterior_theta_param2.append([z_mean2, z_std2, alpha_del])

        ###########################################################################
        # HLSM Decoder
        self.theta1_list = []
        self.theta2_list = []
        self.reconstructions_list = []
        self.posterior_theta_param1_list = []
        self.posterior_theta_param2_list = []
        self.prior_theta_param_list = []
        #Take multiple MC samples
        for k in range(self.S):
            self.theta1 = []
            self.theta2 = []
            self.prior_theta_param = []
            # Downward Inference Pass
            for idx, decoder_layer in enumerate(self.decoder_layers): # l = 1, 2, ..., L-1
                
                if idx == 0:
                    mean_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param1[idx][0] += mean_layer(self.h1[-1][0][0])
                    self.posterior_theta_param2[idx][0] += mean_layer(self.h2[-1][0][0])
                    
                    alpha_gam_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                    self.posterior_theta_param1[idx][2] += alpha_gam_layer(self.h1[-1][2][0])
                    
                    if FLAGS.directed == 1:
                        alpha_del_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                        self.posterior_theta_param2[idx][2] += alpha_del_layer(self.h2[-1][2][0])
                    else:
                        self.posterior_theta_param2[idx][2] = self.posterior_theta_param1[idx][2]
                
                else:
                    mean_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param1[idx][0] += mean_layer(self.theta1[idx - 1][0])
                    self.posterior_theta_param2[idx][0] += mean_layer(self.theta2[idx - 1][0])
                    
                    alpha_gam_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                    self.posterior_theta_param1[idx][2] += alpha_gam_layer(self.theta1[idx - 1][1])
                    
                    if FLAGS.directed == 1:
                        alpha_del_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                        self.posterior_theta_param2[idx][2] += alpha_del_layer(self.theta2[idx - 1][1])
                    else:
                        self.posterior_theta_param2[idx][2] = self.posterior_theta_param1[idx][2]
                
                # latent position
                z1 = sample_normal(self.posterior_theta_param1[idx][0], self.posterior_theta_param1[idx][1])
                z2 = sample_normal(self.posterior_theta_param2[idx][0], self.posterior_theta_param2[idx][1])
                 # node random factors
                alpha_gam_prior = tf.constant(1. / decoder_layer, tf.float32)
                # composing Gamma distribution (with same beta) as Dirichlet distribution
                gamma = sample_gamma(self.posterior_theta_param1[idx][2], tf.constant(FLAGS.beta, tf.float32)) 
                gamma = gamma / (tf.reduce_sum(gamma, axis = 0) + SMALL)
            
                if FLAGS.directed == 1:
                    alpha_del_prior = tf.constant(1. / decoder_layer, tf.float32)
                    # composing Gamma distribution (with same beta) as Dirichlet distribution
                    delta = sample_gamma(self.posterior_theta_param2[idx][2], tf.constant(FLAGS.beta, tf.float32))
                    delta = delta / (tf.reduce_sum(delta, axis = 0) + SMALL)
                else:
                    delta = gamma
                
                self.prior_theta_param.append([alpha_gam_prior, alpha_del_prior])
                self.theta1.append([z1, gamma])
                self.theta2.append([z2, delta])

            output_layer = LSMDecoder(input_dim = self.decoder_layers[-1], num_edges = self.batch_size, act = lambda x: tf.nn.sigmoid(x), logging = self.logging)
            self.reconstructions, self.theta_decoder = output_layer([self.theta1[-1], self.theta2[-1]])
            
            self.theta1_list.append(self.theta1)
            self.theta2_list.append(self.theta2)
            self.reconstructions_list.append(self.reconstructions)
            self.posterior_theta_param1_list.append(self.posterior_theta_param1)
            self.posterior_theta_param2_list.append(self.posterior_theta_param2)
            self.prior_theta_param_list.append(self.prior_theta_param)
