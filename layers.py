from initializations import weight_variable_glorot
import tensorflow as tf
from utils import scaled_distance

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
SMALL = 1e-16

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class FullConnection(Layer):
    def __init__(self, input_dim, output_dim, act, dropout = 0., reuse_name = '', **kwargs):
        super().__init__(**kwargs)
        
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
        
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name = "bias")
        
        self.act = act
        self.dropout = dropout

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, self.dropout)
        x = tf.matmul(x, self.vars['weights']) + self.vars['bias']
        
        output = self.act(x)
        return output

class GraphConvolution(Layer):
    """Degree-gated GNN
    """
    def __init__(self, input_dim, output_dim, act = tf.nn.relu, dropout = 0., bias = False, **kwargs):
        super().__init__(**kwargs)

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights_self'] = weight_variable_glorot(input_dim, output_dim, name = 'weights_self')
            self.vars['weights_nei_out'] = weight_variable_glorot(input_dim, output_dim, name = 'weights_nei_out')
            self.vars['weights_nei_in'] = weight_variable_glorot(input_dim, output_dim, name = 'weights_nei_in')
            
        with tf.compat.v1.variable_scope(self.name + '_merge_vars'):
            self.vars['weight_self'] = weight_variable_glorot(1, 1, name = 'weight_self')
            self.vars['weight_nei_out'] = weight_variable_glorot(1, 1, name = 'weight_nei_out')
            self.vars['weight_nei_in'] = weight_variable_glorot(1, 1, name = 'weight_nei_in')
            if bias:
                self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name = "bias")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout = dropout
        self.bias = bias

    def _call(self, inputs):
        x_self, x_neighbor_out, x_neighbor_in = inputs
        
        x_self = tf.nn.dropout(x_self, self.dropout) # dim: batch_size × K^l
        x_neighbor_out = tf.nn.dropout(x_neighbor_out, self.dropout) # dim: batch_size × num_neighbor × K^l
        x_neighbor_in = tf.nn.dropout(x_neighbor_in, self.dropout) # dim: batch_size × num_neighbor × K^l
        neighbor_out_means = tf.reduce_mean(x_neighbor_out, axis = 1) # dim: batch_size × K^l
        neighbor_in_means = tf.reduce_mean(x_neighbor_in, axis = 1) # dim: batch_size × K^l
        
        # [nodes] x [out_dim]
        from_self = tf.matmul(x_self, self.vars["weights_self"])
        from_nei_out = tf.matmul(neighbor_out_means, self.vars['weights_nei_out'])
        from_nei_in = tf.matmul(neighbor_in_means, self.vars['weights_nei_in'])
        
        output = self.vars['weight_self'] * from_self + self.vars['weight_nei_out'] * from_nei_out + self.vars['weight_nei_in'] * from_nei_in

        # bias
        if self.bias:
            output += self.vars['bias']
        
        return self.act(output)

class LSMDecoder(Layer):
    """Decoder model layer for DLSM."""
    def __init__(self, input_dim, num_edges, act = tf.nn.sigmoid, dropout = 0., **kwargs):
        super().__init__(**kwargs)
                
        self.act = act
        self.dropout = dropout
        self.input_dim = input_dim
        self.n_samples = num_edges
        self.latent_dim = FLAGS.latent_dim
            
        with tf.compat.v1.variable_scope(self.name + '_vars_bias'):
            self.vars['bias'] = tf.Variable(1., name = "bias")
        
        with tf.compat.v1.variable_scope(self.name + '_vars_z'):
            self.vars['transform_z'] = weight_variable_glorot(input_dim, self.latent_dim, name = "transform_z")
            self.vars['transform_z'] = tf.nn.softmax(self.vars['transform_z'], axis = 0)
        
        with tf.compat.v1.variable_scope(self.name + '_vars_gam'):
            self.vars['transform_gam'] = weight_variable_glorot(input_dim, self.latent_dim, name = "transform_gam")
            self.vars['transform_gam'] = tf.nn.softmax(self.vars['transform_gam'], axis = 0) # normalized
            
        if FLAGS.directed == 1:
            with tf.compat.v1.variable_scope(self.name + '_vars_del'):
                self.vars['transform_del'] = weight_variable_glorot(input_dim, self.latent_dim, name = "transform_del")
                self.vars['transform_del'] = tf.nn.softmax(self.vars['transform_del'], axis = 0) # normalized

            with tf.compat.v1.variable_scope(self.name + '_vars_weights'):
                self.vars['weight_gam'] = tf.Variable(0.5, name = "weight_gam")
                self.vars['weight_del'] = tf.Variable(0.5, name = "weight_del")
        else:
            with tf.compat.v1.variable_scope(self.name + '_vars_weights'):
                self.vars['weight_gam'] = tf.Variable(0.5, name = "weight_gam")

    def _call(self, inputs):
        z1 = inputs[0][0]
        z_decoder1 = tf.matmul(z1, self.vars['transform_z'])
        z2 = inputs[1][0]
        z_decoder2 = tf.matmul(z2, self.vars['transform_z'])
        
        gamma = inputs[0][1]
        gamma_decoder = tf.matmul(gamma, self.vars['transform_gam'])
        dist_gam = tf.multiply(self.n_samples, scaled_distance(z_decoder1, z_decoder2, gamma_decoder + SMALL))
        
        if FLAGS.directed == 1:
            delta = inputs[1][1]
            delta_decoder = tf.matmul(delta, self.vars['transform_del'])
            dist_del = tf.multiply(self.n_samples, scaled_distance(z_decoder1, z_decoder2, delta_decoder + SMALL))
            
            x = self.vars['bias'] - self.vars['weight_gam'] * dist_gam - self.vars['weight_del'] * dist_del
        else:
            delta_decoder = gamma_decoder
            x = self.vars['bias'] - self.vars['weight_gam'] * (dist_gam + tf.transpose(dist_gam))

        x = tf.reshape(x, [-1])
        output = self.act(x)

        return output, (z_decoder1, z_decoder2, gamma_decoder, delta_decoder)
    
    def get_weight_matrix(self):
        W = tf.eye(self.input_dim)
        return W
