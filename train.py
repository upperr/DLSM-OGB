#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import time
import os
import tensorflow as tf
import numpy as np
from scipy.stats import mode
import collections
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.cluster import KMeans

from optimizer import Optimizer
from input_data import load_data
from model import DLSM, DLSM_D
from minibatch import EdgeMinibatchIterator

# Settings
tf.compat.v1.disable_v2_behavior()
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
# Experiment settings
flags.DEFINE_string('model', 'dlsm', 'Model to use: dlsm_d')
flags.DEFINE_string('dataset', 'ogbl-citation2', 'OGB Datasets: ogbl- for link prediction, ogbn- for community detection')
flags.DEFINE_integer('directed', 1, 'Whether the network is directed (1) or not (0).')
flags.DEFINE_integer('link_prediction', 0, 'Conduct link prediction')
flags.DEFINE_integer('community_detection', 0, 'Conduct community detection')
flags.DEFINE_integer('comm_least_size', 0, 'Least number of nodes for each community (30 for email)')
flags.DEFINE_integer('batch_size', 256, 'minibatch size.')
flags.DEFINE_integer('validate_batch_size', 128, "how many edges per validation iteration.")
flags.DEFINE_integer('epochs', 50, 'Max number of epochs to train. Training may stop early if validation-error is not decreasing')
flags.DEFINE_integer('early_stopping', 10, 'Number epochs to train after last best validation')
flags.DEFINE_integer('validate_iter', 500, "how often to run a validation minibatch.")
flags.DEFINE_string('gpu_to_use', '0', 'Which GPU to use. Leave blank to use None')
flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
# Model settings
flags.DEFINE_string('encoder', '32_64_128', 'Number of units in encoder layers')
flags.DEFINE_string('decoder', '50_100', 'Number of units in decoder layers')
flags.DEFINE_integer('latent_dim', 50, 'Dimension of latent space (readout layer)')
flags.DEFINE_string('num_neighbors', '10_5_3', 'Number of nodes sampled at each layer. Connect numbers using _.')
flags.DEFINE_integer('max_degree', 50, 'maximum node degree.')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('use_kl_warmup', 1, 'Use a linearly increasing KL [0-1] coefficient -- see wu_beta in optimization.py')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('beta', 1., 'Posterior beta for Gamma')
flags.DEFINE_float('v0', 0.9, 'Prior parameter for steak-breaking IBP')
flags.DEFINE_float('temp_prior', 0.5, 'Prior temperature for concrete distribution')
flags.DEFINE_float('temp_post', 1., 'Posterior temperature for concrete distribution')
flags.DEFINE_integer('mc_samples', 1, 'No. of MC samples for calculating gradients')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_to_use

save_dir =  "data/" + model_str + "/" + dataset_str + "/" + FLAGS.encoder + "/"
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, S = 2, size = None):

    feed_dict_val, labels = minibatch_iter.val_feed_dict(size)
    
    val_preds = np.zeros(labels.shape)
    for s in range(S):
        outs = sess.run([model.reconstructions], feed_dict = feed_dict_val)
        val_preds += outs[0]
    val_preds = val_preds / S
    # Get ROC score and average precision
    roc_score = roc_auc_score(labels, val_preds)
    ap_score = average_precision_score(labels, val_preds)

    return roc_score, ap_score

def incremental_evaluate(sess, model, minibatch_iter, size, S = 5):

    test_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_test, batch_labels, finished = minibatch_iter.incremental_test_feed_dict(size, iter_num)
        batch_test_preds = np.zeros(batch_labels.shape)
        for s in range(S):
            outs = sess.run([model.reconstructions], feed_dict = feed_dict_test)
            batch_test_preds += outs[0]
        batch_test_preds = batch_test_preds / S
        test_preds.append(batch_test_preds)
        labels.append(batch_labels)
        iter_num += 1
        
    test_preds = np.hstack(test_preds)
    labels = np.hstack(labels)
    roc_score = roc_auc_score(labels, test_preds)
    ap_score = average_precision_score(labels, test_preds)
    
    return roc_score, ap_score

# create_model 
def create_model(placeholders, neighbors_out, neighbors_in, features):
    # Create model
    if model_str == 'dlsm':
        model = DLSM(placeholders,
                     neighbors_out = neighbors_out,
                     neighbors_in = neighbors_in,
                     features = features,
                     mc_samples = FLAGS.mc_samples,
                     identity_dim = FLAGS.identity_dim)
    elif model_str == 'dlsm_d':
        model = DLSM_D(placeholders,
                       neighbors_out = neighbors_out,
                       neighbors_in = neighbors_in,
                       features = features,
                       mc_samples = FLAGS.mc_samples,
                       identity_dim = FLAGS.identity_dim)
    else:
        raise NameError('No model named ' + model_str + '!')
        
    # Optimizer
    with tf.compat.v1.name_scope('optimizer'):
        opt = Optimizer(labels = placeholders['labels'],
                        model = model,
                        epoch = placeholders['epoch'],
                        model_str = model_str)
    return model, opt

def train(placeholders, model, opt, sess, minibatch, neighbors_out_ph, neighbors_in_ph, neighbors_train, neighbors, name = "link_prediction"):
    
    sess.run(tf.compat.v1.global_variables_initializer(), feed_dict = {neighbors_out_ph: neighbors_train['out'],
                                                                       neighbors_in_ph: neighbors_train['in']})
    saver = tf.compat.v1.train.Saver(max_to_keep = 0)

    best_validation = -float('inf')
    last_best_epoch = 0
    if FLAGS.community_detection:
        z = np.zeros([neighbors_train['out'].shape[0] - 1, FLAGS.latent_dim])
        name = 'community_detection'
    else:
        z = None
    
    train_neighbors_out = tf.compat.v1.assign(model.neighbors_out, neighbors_train['out'])
    train_neighbors_in = tf.compat.v1.assign(model.neighbors_in, neighbors_train['in'])
    if FLAGS.link_prediction:
        val_neighbors_out = tf.compat.v1.assign(model.neighbors_out, neighbors['out'])
        val_neighbors_in = tf.compat.v1.assign(model.neighbors_in, neighbors['in'])
    
    # Train model
    time_start = time.time()
    for epoch in range(FLAGS.epochs):

        minibatch.shuffle() 
        iteration = 0
        print('Epoch: %04d' % (epoch + 1))
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict() # batch_size, batch1, batch2
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            feed_dict.update({placeholders['epoch']: epoch})
            # Training step
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl, model.nodes1, model.nodes2, model.theta_decoder], feed_dict = feed_dict)
            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]
            kl = outs[3]
            nodes_source = outs[4]
            nodes_target = outs[5]
            z_source = outs[6][0]
            z_target = outs[6][1]
            if FLAGS.community_detection:
                z[nodes_source, :] = z_source
                z[nodes_target, :] = z_target
            
            if iteration % FLAGS.validate_iter == 0:
                if FLAGS.link_prediction:
                    # Validation
                    sess.run([val_neighbors_out.op, val_neighbors_in.op])
                    val_auc, val_ap = evaluate(sess, model, minibatch, size = FLAGS.validate_batch_size)
                    sess.run([train_neighbors_out.op, train_neighbors_in.op])
                    # Print results                
                    t = time.time() - time_start
                    time_start = time.time()
                    print("Iter:", '%04d' % iteration, 
                          "train_loss=", "{:.3f}".format(avg_cost),
                          "train_kl=", "{:.3f}".format(kl),
                          "train_acc=", "{:.3f}".format(avg_accuracy), 
                          "val_auc=", "{:.3f}".format(val_auc), 
                          "val_ap=", "{:.3f}".format(val_ap),
                          "time=", "{:.5f}".format(t))
            
                    if val_auc > best_validation:
                        # save model
                        print ('Saving model')
                        #saver.save(sess = sess, save_path = save_dir + name)
                        best_validation = val_auc
                        best_epoch = epoch
                        last_best_epoch = 0
                    
                elif FLAGS.community_detection:
                    # Print results
                    t = time.time() - time_start
                    time_start = time.time()
                    print("Iter:", '%04d' % iteration, 
                          "train_loss=", "{:.3f}".format(avg_cost),
                          "train_kl=", "{:.3f}".format(kl),
                          "train_acc=", "{:.3f}".format(avg_accuracy), 
                          "time=", "{:.5f}".format(t))
            
                    if -avg_cost > best_validation:
                        # save model
                        print ('Saving model')
                        saver.save(sess = sess, save_path = save_dir + name)
                        best_validation = -avg_cost
                        last_best_epoch = 0
                
            iteration += 1
        
        if last_best_epoch > FLAGS.early_stopping:
            break
        else:
            last_best_epoch += 1
    
    print("Optimization Finished!")
    
    if FLAGS.link_prediction:
        print('Validation AUC Max: {:.3f} at Epoch: {:04d}'.format(best_validation, best_epoch))
        #saver.restore(sess = sess, save_path = save_dir + name)
        print ('Model restored')
        # Testing
        sess.run([val_neighbors_out.op, val_neighbors_in.op])
        test_auc, test_ap = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
        print('Test AUC score: {:.3f}'.format(test_auc))
        print('Test AP score: {:.3f}'.format(test_ap))
        with open("link_prediction_results.txt", "w") as fp:
            fp.write("AUC={:.5f} AP={:.5f}".format(test_auc, test_ap))
    
    return z

def main():
    
    if (FLAGS.link_prediction and dataset_str.startswith('ogbn')) or (FLAGS.community_detection and dataset_str.startswith('ogbl')):
        raise Exception('Dataset dose not match task!')
            
    print("Loading training data..")
    train_edges, val_edges, test_edges, labels, features, nei_train, nei = load_data(dataset_str, FLAGS.max_degree)
    print(nei_train['out'].shape)
    print(nei_train['in'].shape)
    if isinstance(features, np.ndarray):
        features = np.vstack([features, np.zeros([1, features.shape[1]])]) # use zeros as inputs for nodes without out- or in-neighbors
    print("Done loading training data!")

    print ("Model is " + model_str)

    # Define placeholders
    placeholders = {
        'labels': tf.compat.v1.placeholder(tf.float32),
        'batch_nodes_source' : tf.compat.v1.placeholder(tf.int32),
        'batch_nodes_target' : tf.compat.v1.placeholder(tf.int32),
        'batch_size' : tf.compat.v1.placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'epoch': tf.compat.v1.placeholder(tf.int32)
    }
    
    print('Constructing minibatch iterator..')
    minibatch = EdgeMinibatchIterator(train_edges, 
                                      val_edges, 
                                      test_edges,
                                      placeholders,
                                      batch_size = FLAGS.batch_size)
    print('Done constructing Minibatch iterator!')
    del train_edges, val_edges, test_edges
    
    nei_out_train_ph = tf.compat.v1.placeholder(tf.int32, shape = nei_train['out'].shape)
    nei_out_train = tf.Variable(nei_out_train_ph, trainable = False, name = "neighbors_out_train")
    nei_in_train_ph = tf.compat.v1.placeholder(tf.int32, shape = nei_train['in'].shape)
    nei_in_train = tf.Variable(nei_in_train_ph, trainable = False, name = "neighbors_in_train")

    model, opt = create_model(placeholders, nei_out_train, nei_in_train, features)
    config = tf.compat.v1.ConfigProto(log_device_placement = FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.compat.v1.Session(config = config)
    
    emb = train(placeholders, model, opt, sess, minibatch, nei_out_train_ph, nei_in_train_ph, nei_train, nei)
    
    if FLAGS.community_detection:
        label_count = collections.Counter(labels)
        # eliminate small communities
        comm_keep = np.array(list(label_count.keys()))[np.array(list(label_count.values())) > FLAGS.comm_least_size]
        z_keep = emb[np.in1d(labels, comm_keep), :]
        labels_keep = labels[np.in1d(labels, comm_keep)]
        num_classes = comm_keep.shape[0]
        print('Keep ' + str(num_classes) + ' communities.')

        km = KMeans(n_clusters = num_classes)
        print('Perform K-means for community detection with the learned embeddings..')
        clusters = km.fit_predict(z_keep) # cluster using K-means
        preds = np.zeros_like(clusters)
        for i in range(num_classes):
            # get a bool index matrix of the i-th clustering community
            mask = (clusters == i) 
            # find the most linkely community in the real labels
            preds[mask] = mode(labels_keep[mask])[0]
        
        f1_macro = f1_score(labels_keep, preds, average = 'macro')
        f1_micro = f1_score(labels_keep, preds, average = 'micro')
        print('Macro F1-score: ' + str(f1_macro))
        print('Micro F1-score: ' + str(f1_micro))
        with open("community_detection_results.txt", "w") as fp:
            fp.write("Macro F1-score={:.5f} Micro F1-score={:.5f}".format(f1_macro, f1_micro))

if __name__ == '__main__':
    main()
