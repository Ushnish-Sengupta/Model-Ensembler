import tensorflow as tf
print(tf.__version__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.utils import shuffle
import os
import pickle as pkl

import datetime

# # Make reproducible
seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)


def report_on_percentiles(y, y_pred, y_std):

    n = len(y.ravel())

    n1 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 1)
    n2 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 2)
    n3 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 3)
    print('Using {} data points'.format(n))
    print('{} within 1 std'.format(100 * n1 / n))
    print('{} within 2 std'.format(100 * n2 / n))
    print('{} within 3 std'.format(100 * n3 / n))

    return


def recube(in_array):

    lat_len = 64
    lon_len = 128
    time_len = 31 * 12

    output = np.zeros([time_len, lat_len, lon_len])

    for t in range(time_len):
        output[t,:,:] = in_array[lat_len * lon_len * (t): lat_len * lon_len * (t+1)].reshape([lat_len, lon_len])
    
    return output


# Setup for saving


save_path = 'savedir/{}'


# Load data

df = pd.read_pickle('refC1SD_MINMAX_64x128_raw.pkl')

# Apply coordinate mapping lat,lon -> x,y,z
lon = df['lon'] * np.pi / 180
lat = df['lat'] * np.pi / 180
x = np.cos(lat) * np.cos(lon)
y = np.cos(lat) * np.sin(lon)
z = np.sin(lat)

# Apply coordinate mapping month_number -> x_mon, y_mon
rads = (df['mon_num'] * 360/12) * (np.pi / 180)
x_mon = np.sin(rads)
y_mon = np.cos(rads)

# min-max scale months (months since Jan 1980)
mons_scaled = 2 * (df['mons'] - df['mons'].min())/(df['mons'].max() - df['mons'].min()) - 1

# Remove old coords and add new mapped coords from/to dataframe
df = df.drop(['lat', 'lon', 'mon_num', 'mons'], axis=1)
df['x'] = x
df['y'] = y
df['z'] = z
df['x_mon'] = x_mon
df['y_mon'] = y_mon
df['mons'] = mons_scaled

# Apply min-max scaling to each model and observations
y_min = df['obs_toz'].min()
y_max = df['obs_toz'].max()
for i in range(16): # 15 models and 1 obs
    mdl = df[df.columns[i]]
    df[df.columns[i]] = 2 * (mdl - mdl.min())/(mdl.max() - mdl.min()) - 1
    
# Apply coordinate scaling
df['x'] = df['x'] * 2
df['y'] = df['y'] * 2
df['z'] = df['z'] * 2

df['x_mon'] = df['x_mon'] * 2
df['y_mon'] = df['y_mon'] * 2
df['mons'] = df['mons'] * 1.0

cols_to_drop = ['has_obs', 'train_mask', 'test_mask', 'interp_mask', 'extrap_mask']
# Maintain shuffling with seed
df_train = df[df['train_mask']].drop(cols_to_drop, axis=1)
df_test = df[df['test_mask']].drop(cols_to_drop, axis=1)
df_interp = df[df['interp_mask']].drop(cols_to_drop, axis=1)
df_extrap = df[df['extrap_mask']].drop(cols_to_drop, axis=1)
n_obs = np.sum(df['has_obs'].values)

print('Training on {:.1f}%'.format(100 * len(df_train)/n_obs))
print('Testing on {:.1f}%'.format(100 * len(df_test)/n_obs))
print('Validation (interpolation) on {:.1f}%'.format(100 * len(df_interp)/n_obs))
print('Validation (extrapolation) on {:.1f}%'.format(100 * len(df_extrap)/n_obs))

# In sample training
X_train = df_train.drop(['obs_toz'],axis=1).values
y_train = df_train['obs_toz'].values.reshape(-1,1)

# The in sample testing - this is not used for training
X_test = df_test.drop(['obs_toz'],axis=1).values
y_test = df_test['obs_toz'].values.reshape(-1,1)

# For all time
X_at = df.drop(['obs_toz'],axis=1).drop(cols_to_drop, axis=1).values
y_at = df.drop(cols_to_drop, axis=1)['obs_toz'].values.reshape(-1,1)


# NN set up
tf.reset_default_graph()

num_models = 15

# Bias mean is assumed 0
bias_mean = 0.00
bias_std = 0.03

# prior on the noise 
noise_mean = 0.015
noise_std = 0.003

# hyperparameters
n = X_train.shape[0]
x_dim = X_train.shape[1]
alpha_dim = x_dim - num_models
y_dim = y_train.shape[1]

n_ensembles = 65
hidden_size = 500
init_stddev_1_w =  np.sqrt(3.0/(alpha_dim))
init_stddev_1_b = init_stddev_1_w
init_stddev_2_w =  (1.26)/np.sqrt(hidden_size)
init_stddev_2_b = init_stddev_2_w
init_stddev_3_w = (1.26*bias_std)/np.sqrt(hidden_size)
init_stddev_noise_w = (1.26*noise_std)/np.sqrt(hidden_size)

lambda_anchor = 1.0/(np.array([init_stddev_1_w,init_stddev_1_b,init_stddev_2_w,init_stddev_2_b,init_stddev_3_w,init_stddev_noise_w])**2)

n_epochs = 125000
batch_size = 25000
learning_rate = 0.00003

# NN class
class NN():
    def __init__(self, x_dim, y_dim, hidden_size, init_stddev_1_w, init_stddev_1_b, init_stddev_2_w, init_stddev_2_b, init_stddev_3_w, init_stddev_noise_w, learning_rate, model_bias_from_layer):
        # setting up as for a usual NN
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # set up NN
        self.inputs = tf.placeholder(tf.float32, [None, x_dim], name='inputs')
        self.modelpred = self.inputs[:, :num_models]
        self.spacetime = self.inputs[:, num_models: num_models + alpha_dim]
        self.area_weights = self.inputs[:, -1]
        self.y_target = tf.placeholder(tf.float32, [None, y_dim], name='target')
        
        self.layer_1_w = tf.layers.Dense(hidden_size, activation=tf.nn.tanh,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_1_w),
                                         bias_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_1_b))
        self.layer_1 = self.layer_1_w.apply(self.spacetime)
        self.layer_2_w = tf.layers.Dense(num_models, activation=None,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_2_w),
                                         bias_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_2_b))
        self.layer_2 = self.layer_2_w.apply(self.layer_1)

        self.model_coeff = tf.nn.softmax(self.layer_2)

        self.modelbias_w = tf.layers.Dense(y_dim, activation=None, use_bias=False,
                                           kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_3_w))
        if model_bias_from_layer == 1:
            self.modelbias = self.modelbias_w.apply(self.layer_1)
        elif model_bias_from_layer == 2:
            self.modelbias = self.modelbias_w.apply(self.layer_2)
            
        self.output = tf.reduce_sum(self.model_coeff * self.modelpred, axis=1) + tf.reshape(self.modelbias, [-1])
        
        self.noise_w = tf.layers.Dense(self.y_dim, activation=None, use_bias=False,
                                       kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_noise_w))
        self.noise_pred = self.noise_w.apply(self.layer_1)

        # set up loss and optimiser - we'll modify this later with anchoring regularisation
        self.opt_method = tf.train.AdamOptimizer(self.learning_rate)
        self.noise_sq = tf.square(self.noise_pred + noise_mean)[:,0] + 1e-6 
        self.err_sq = tf.reshape(tf.square(self.y_target[:,0] - self.output), [-1])
        num_data_inv = tf.cast(tf.divide(1, tf.shape(self.inputs)[0]), dtype=tf.float32)

        self.mse_ = num_data_inv * tf.reduce_sum(self.err_sq) 
        self.loss_ = num_data_inv * (tf.reduce_sum(tf.divide(self.err_sq, self.noise_sq)) + tf.reduce_sum(tf.log(self.noise_sq)))
        self.optimizer = self.opt_method.minimize(self.loss_)

        return


    def get_weights(self, sess):
        '''method to return current params'''
        ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.layer_2_w.kernel, self.layer_2_w.bias, self.modelbias_w.kernel, self.noise_w.kernel]
        w1, b1, w2, b2, w3, wn = sess.run(ops)
        return w1, b1, w2, b2, w3, wn

    def anchor(self, sess, lambda_anchor):
        '''regularise around initialised parameters'''
        w1, b1, w2, b2, w3, wn = self.get_weights(sess)

        # get initial params
        self.w1_init, self.b1_init, self.w2_init, self.b2_init, self.w3_init, self.wn_init = w1, b1, w2, b2, w3, wn
        loss_anchor = lambda_anchor[0]*tf.reduce_sum(tf.square(self.w1_init - self.layer_1_w.kernel))
        loss_anchor += lambda_anchor[1]*tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))
        loss_anchor += lambda_anchor[2]*tf.reduce_sum(tf.square(self.w2_init - self.layer_2_w.kernel))
        loss_anchor += lambda_anchor[3]*tf.reduce_sum(tf.square(self.b2_init - self.layer_2_w.bias))
        loss_anchor += lambda_anchor[4]*tf.reduce_sum(tf.square(self.w3_init - self.modelbias_w.kernel))
        loss_anchor += lambda_anchor[5]*tf.reduce_sum(tf.square(self.wn_init - self.noise_w.kernel)) # new param

        self.loss_anchor = tf.cast(1.0/X_train.shape[0], dtype=tf.float32) * loss_anchor
        
        # combine with original loss
        self.loss_ = self.loss_ + tf.cast(1.0/X_train.shape[0], dtype=tf.float32) * loss_anchor
        self.optimizer = self.opt_method.minimize(self.loss_)
        return


    def predict(self, x, sess):
        '''predict method'''
        feed = {self.inputs: x}
        y_pred = sess.run(self.output, feed_dict=feed)
        return y_pred
    
    def get_noise_sq(self, x, sess):
        '''get noise squared method'''
        feed = {self.inputs: x}
        noise_sq = sess.run(self.noise_sq, feed_dict=feed)
        return noise_sq

    def get_alphas(self, x, sess):
        feed = {self.inputs: x}
        alpha = sess.run(self.model_coeff, feed_dict=feed)
        return alpha
    
    def get_betas(self, x, sess):
        feed = {self.inputs: x}
        beta = sess.run(self.modelbias, feed_dict=feed)
        return beta

    def get_alpha_w(self, x, sess):
      feed = {self.inputs: x}
      alpha_w = sess.run(self.layer_2, feed_dict=feed)
      return alpha_w

    def get_w1(self, x, sess):
      feed = {self.inputs: x}
      w1 = sess.run(self.layer_1, feed_dict=feed)
      return w1

def fn_predict_ensemble(NNs,X_train):
    y_pred=[]
    y_pred_noise_sq=[]
    for ens in range(0,n_ensembles):
        y_pred.append(NNs[ens].predict(X_train, sess))
        y_pred_noise_sq.append(NNs[ens].get_noise_sq(X_train, sess))
    y_preds_train = np.array(y_pred)
    y_preds_noisesq_train = np.array(y_pred_noise_sq)
    y_preds_mu_train = np.mean(y_preds_train,axis=0)
    y_preds_std_train_epi = np.std(y_preds_train,axis=0)
    y_preds_std_train = np.sqrt(np.mean((y_preds_noisesq_train + np.square(y_preds_train)), axis = 0) - np.square(y_preds_mu_train)) #add predicted aleatoric noise
    return y_preds_train, y_preds_mu_train, y_preds_std_train, y_preds_std_train_epi, y_preds_noisesq_train

def get_alphas(NNs, X_train):
    alphas = []
    for ens in range(0,n_ensembles):
        alphas.append(NNs[ens].get_alphas(X_train, sess))
    return alphas


def get_betas(NNs, X_train):
    betas = []
    for ens in range(0,n_ensembles):
        betas.append(NNs[ens].get_betas(X_train, sess))
    return betas

def get_layer2_output(NNs, X_train):
  alpha_w = []
  for ens in range(0,n_ensembles):
    alpha_w.append(NNs[ens].get_alpha_w(X_train, sess))
  return alpha_w

def get_layer1_output(NNs, X_train):
  w1 = []
  for ens in range(0,n_ensembles):
    w1.append(NNs[ens].get_w1(X_train, sess))
  return w1


# Initialise the NNs


NNs=[]

sess = tf.Session()
init_weights = []

# loop to initialise all ensemble members
for ens in range(0,n_ensembles):
    NNs.append(NN(x_dim, y_dim, hidden_size, init_stddev_1_w, init_stddev_1_b, init_stddev_2_w, init_stddev_2_b, init_stddev_3_w, init_stddev_noise_w,
                  learning_rate, 1))
    # initialise only unitialized variables - stops overwriting ensembles already created
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
      
    # do regularisation now that we've created initialisations
    NNs[ens].anchor(sess, lambda_anchor)


# Check Priors


X_train_short = X_train[:10000]

y_preds_train, y_preds_mu_train, y_preds_std_train, y_preds_std_train_epi, y_preds_noisesq_train = fn_predict_ensemble(NNs,X_train_short)
plt.figure(figsize=(8,8))
plt.errorbar(y_train[:10000], y_preds_mu_train[:10000], yerr=(y_preds_std_train*1)[:10000],linewidth = 0.25, color = 'gray', ms=2,mfc='red',mec='black', fmt='o')
plt.plot(np.arange(np.min(y_train[:10000]), np.max(y_train[:10000]), 0.01), np.arange(np.min(y_train[:10000]), np.max(y_train[:10000]), 0.01), linewidth = 2, linestyle = 'dashed',zorder = 100)
plt.xlabel('True concentration')
plt.ylabel('Predicted concentration')
plt.savefig('Prior_predictive_check.png')

y_preds_train, y_preds_mu_train, y_preds_std_train, y_preds_std_train_epi, y_preds_noisesq_train = fn_predict_ensemble(NNs,X_train)

# Alphas
alphas = np.array(get_alphas(NNs, X_train_short))
print('Alpha mean should be: {}'.format(1/num_models))
print('Alpha mean is: {}'.format(np.mean(np.array(alphas).ravel())))
print('Alpha std should be: {}'.format(np.sqrt((1/(1 + num_models)) * (1/num_models)*(1-(1/num_models)))))
print('Alpha std is: {}'.format(np.mean(np.std(np.array(alphas), axis=0).ravel())))
report_on_percentiles(alphas, np.array(1/num_models), np.mean(np.std(np.array(alphas), axis=0).ravel()))
print('')

# Beta
beta = np.array(get_betas(NNs, X_train))
print('Beta mean should be: {}'.format(bias_mean))
print('Beta mean is: {}'.format(np.mean(beta.ravel())))
print('Beta std should be: {}'.format(bias_std))
print('Beta std is: {}'.format(np.std(beta.ravel())))
# report_on_percentiles(beta, bias_mean, np.std(beta.ravel()))
print('')

# Network weights
print('For the layers')
w1 = np.array(get_layer1_output(NNs, X_train_short))
alpha_w = np.array(get_layer2_output(NNs, X_train_short))
print('Layer 1 mean: {}'.format(np.mean(w1.ravel())))
print('Layer 2 mean: {}'.format(np.mean(alpha_w.ravel())))
print('Layer 1 Std: {}'.format(np.mean(np.std(w1, axis=0).ravel())))
print('Layer 2 Std: {}'.format(np.mean(np.std(alpha_w, axis=0).ravel())))
print('')

# Noise
print('For noise')
pred_noise = np.sqrt(np.array([NN.get_noise_sq(X_train, sess) for NN in NNs]))
report_on_percentiles(pred_noise, np.array(noise_mean), np.array(noise_std))
print('')

### Untrained output
print('For the predictions')
report_on_percentiles(y_train, y_preds_mu_train, y_preds_std_train)


# Training

l_s = []
m_s = []
a_s = []

for ens in range(0,n_ensembles):
    ep_ = 0
    losses = []
    mses = []
    anchs = []
    print('NN:',ens + 1)
    while ep_ < n_epochs:
        if (ep_ % 50 == 0):
            X_train, y_train = shuffle(X_train, y_train, random_state = ep_)

        ep_ += 1
        for j in range(int(n/batch_size)): #minibatch training loop
            feed_b = {}
            feed_b[NNs[ens].inputs] = X_train[j*batch_size:(j+1)*batch_size, :]
            feed_b[NNs[ens].y_target] = y_train[j*batch_size:(j+1)*batch_size, :]
            blank = sess.run(NNs[ens].optimizer, feed_dict=feed_b)
        if (ep_ % 250) == 0: 
            feed_b = {}
            feed_b[NNs[ens].inputs] = X_train
            feed_b[NNs[ens].y_target] = y_train
            loss_mse = sess.run(NNs[ens].mse_, feed_dict=feed_b)
            loss_anch = sess.run(NNs[ens].loss_, feed_dict=feed_b)
            loss_anch_term = sess.run(NNs[ens].loss_anchor, feed_dict=feed_b)
            losses.append(loss_anch)
            mses.append(loss_mse)
            anchs.append(loss_anch_term)
        if (ep_ % 250 == 0):
            print('epoch:' + str(ep_) + ' at ' + str(datetime.datetime.now()))
            print(', rmse_', np.round(np.sqrt(loss_mse),5), ', loss_anch', np.round(loss_anch,5), ', anch_term', np.round(loss_anch_term,5))
    l_s.append(losses)
    m_s.append(mses)
    a_s.append(anchs)

    # If saving weights      
    weight = NNs[ens].get_weights(sess)
    pkl.dump(weight, open(save_path.format('weights{}.pkl'.format(ens)), 'wb'))



plt.figure()
plt.plot(np.array(l_s).T)
plt.title('anchored loss')
plt.savefig('Anchored_Loss.png')

plt.figure()
plt.plot(np.array(m_s).T)
plt.title('MSE')
plt.savefig('MSE.png')

plt.figure()
plt.plot(np.array(a_s).T)
plt.title('anchoring term')
plt.show('Anchoring.png')


# Make predictions

# The in interpolation validation - not used for testing
X_interp = df_interp.drop(['obs_toz'],axis=1).values
y_interp = df_interp['obs_toz'].values.reshape(-1,1)

# The in extrapolation validation - not used for testing
X_extrap = df_extrap.drop(['obs_toz'],axis=1).values
y_extrap = df_extrap['obs_toz'].values.reshape(-1,1)

#use trained NN ensemble to generate predictions
y_preds_train, y_preds_mu_train, y_preds_std_train,  y_preds_std_train_epi, y_preds_noisesq_train = fn_predict_ensemble(NNs,X_train)
y_preds_test, y_preds_mu_test, y_preds_std_test,  y_preds_std_test_epi, y_preds_noisesq_test = fn_predict_ensemble(NNs,X_test)
y_preds_extrap, y_preds_mu_extrap, y_preds_std_extrap,  y_preds_std_extrap_epi, y_preds_noisesq_extrap = fn_predict_ensemble(NNs,X_extrap)
y_preds_interp, y_preds_mu_interp, y_preds_std_interp,  y_preds_std_interp_epi, y_preds_noisesq_interp = fn_predict_ensemble(NNs,X_interp)


# Look at RMSEs and NLLs


print('Train RMSE: {}'.format(np.sqrt(np.mean(np.square(y_preds_mu_train.ravel() - y_train.ravel())))))
print('Test RMSE: {}'.format(np.sqrt(np.mean(np.square(y_preds_mu_test.ravel() - y_test.ravel())))))
print('Interp RMSE: {}'.format(np.sqrt(np.mean(np.square(y_preds_mu_interp.ravel() - y_interp.ravel())))))
print('Extrap RMSE: {}'.format(np.sqrt(np.mean(np.square(y_preds_mu_extrap.ravel() - y_extrap.ravel())))))

print('Train NLL: {}'.format(np.mean(0.5*((((y_preds_mu_train.ravel() - y_train.ravel())**2)/((y_preds_std_train.ravel()**2)) + np.log(y_preds_std_train.ravel()**2) + np.log(2*np.pi))))))
print('Test NLL: {}'.format(np.mean(0.5*((((y_preds_mu_test.ravel() - y_test.ravel())**2)/((y_preds_std_test.ravel()**2)) + np.log(y_preds_std_test.ravel()**2) + np.log(2*np.pi))))))
print('Interp NLL: {}'.format(np.mean(0.5*((((y_preds_mu_interp.ravel() - y_interp.ravel())**2)/((y_preds_std_interp.ravel()**2)) + np.log(y_preds_std_interp.ravel()**2) + np.log(2*np.pi))))))
print('Extrap NLL: {}'.format(np.mean(0.5*((((y_preds_mu_extrap.ravel() - y_extrap.ravel())**2)/((y_preds_std_extrap.ravel()**2)) + np.log(y_preds_std_extrap.ravel()**2) + np.log(2*np.pi))))))

X_interp_NP = df_interp[df_interp['z'] > 1.75].drop(['obs_toz'],axis=1).values
X_interp_SP = df_interp[df_interp['z'] < -1.75].drop(['obs_toz'],axis=1).values
X_interp_trop = df_interp[(-0.68< df_interp['z']) & (df_interp['z'] < 0.68)].drop(['obs_toz'],axis=1).values
X_interp_NML = df_interp[(1.15 < df_interp['z']) & (df_interp['z'] < 1.75)].drop(['obs_toz'],axis=1).values
X_interp_SML = df_interp[(-1.75< df_interp['z']) & (df_interp['z'] < -1.15)].drop(['obs_toz'],axis=1).values

 

y_interp_NP = df_interp[df_interp['z'] > 1.75]['obs_toz'].values.reshape(-1,1)
y_interp_SP = df_interp[df_interp['z'] < -1.75]['obs_toz'].values.reshape(-1,1)
y_interp_trop = df_interp[(-0.68 < df_interp['z']) & (df_interp['z'] < 0.68)]['obs_toz'].values.reshape(-1,1)
y_interp_NML = df_interp[(1.15 < df_interp['z']) & (df_interp['z'] < 1.75)]['obs_toz'].values.reshape(-1,1)
y_interp_SML = df_interp[(-1.75< df_interp['z']) & (df_interp['z'] < -1.15)]['obs_toz'].values.reshape(-1,1)

y_preds_interp_NP, y_preds_mu_interp_NP, y_preds_std_interp_NP,  y_preds_std_interp_NP_epi, y_preds_noisesq_interp_NP = fn_predict_ensemble(NNs,X_interp_NP)
y_preds_interp_SP, y_preds_mu_interp_SP, y_preds_std_interp_SP,  y_preds_std_interp_SP_epi, y_preds_noisesq_interp_SP = fn_predict_ensemble(NNs,X_interp_SP)
y_preds_interp_trop, y_preds_mu_interp_trop, y_preds_std_interp_trop,  y_preds_std_interp_trop_epi, y_preds_noisesq_interp_trop = fn_predict_ensemble(NNs,X_interp_trop)

print('NP RMSE: {}'.format(np.sqrt(np.mean(np.square(y_preds_mu_interp_NP.ravel() - y_interp_NP.ravel())))))
print('SP RMSE: {}'.format(np.sqrt(np.mean(np.square(y_preds_mu_interp_SP.ravel() - y_interp_SP.ravel())))))
print('Tropics RMSE: {}'.format(np.sqrt(np.mean(np.square(y_preds_mu_interp_trop.ravel() - y_interp_trop.ravel())))))

print('NP NLL: {}'.format(np.mean(0.5*((((y_preds_mu_interp_NP.ravel() - y_interp_NP.ravel())**2)/((y_preds_std_interp_NP.ravel()**2)) + np.log(y_preds_std_interp_NP.ravel()**2) + np.log(2*np.pi))))))
print('SP NLL: {}'.format(np.mean(0.5*((((y_preds_mu_interp_SP.ravel() - y_interp_SP.ravel())**2)/((y_preds_std_interp_SP.ravel()**2)) + np.log(y_preds_std_interp_SP.ravel()**2) + np.log(2*np.pi))))))
print('Tropics NLL: {}'.format(np.mean(0.5*((((y_preds_mu_interp_trop.ravel() - y_interp_trop.ravel())**2)/((y_preds_std_interp_trop.ravel()**2)) + np.log(y_preds_std_interp_trop.ravel()**2) + np.log(2*np.pi))))))


# Compare the distribution of the errors

plt.figure(figsize=(12,6))

plt.subplot(2, 3, 1)
plt.hist((y_train.ravel() - y_preds_mu_train.ravel())/y_preds_std_train.ravel() , bins=20)
plt.title('Train')

plt.subplot(2, 3, 2)
plt.hist((y_test.ravel() - y_preds_mu_test.ravel())/y_preds_std_test.ravel() , bins=20)
plt.title('Test')

plt.subplot(2, 3, 3)
plt.hist((y_extrap.ravel() - y_preds_mu_extrap.ravel())/y_preds_std_extrap.ravel() , bins=20)
plt.title('Extrap')

plt.subplot(2, 3, 4)
plt.hist((y_interp.ravel() - y_preds_mu_interp.ravel())/y_preds_std_interp.ravel() , bins=20)
plt.title('Interp - total')

plt.subplot(2, 3, 5)
plt.hist((y_interp_SP.ravel() - y_preds_mu_interp_SP.ravel())/y_preds_std_interp_SP.ravel() , bins=20)
plt.title('Interp - SP')

plt.subplot(2, 3, 6)
plt.hist((y_interp_trop.ravel() - y_preds_mu_interp_trop.ravel())/y_preds_std_interp_trop.ravel() , bins=20)
plt.title('Interp - tropics')

plt.savefig('Histogram_Errors.png')




print('For train')
report_on_percentiles(y_train, y_preds_mu_train, y_preds_std_train)
print('For test')
report_on_percentiles(y_test, y_preds_mu_test, y_preds_std_test)
print('For Extrap')
report_on_percentiles(y_extrap, y_preds_mu_extrap, y_preds_std_extrap)
print('For interp')
report_on_percentiles(y_interp, y_preds_mu_interp, y_preds_std_interp)

report_on_percentiles(y_interp_NP, y_preds_mu_interp_NP, y_preds_std_interp_NP)
report_on_percentiles(y_interp_SP, y_preds_mu_interp_SP, y_preds_std_interp_SP)
report_on_percentiles(y_interp_trop, y_preds_mu_interp_trop, y_preds_std_interp_trop)


# Model Coefficients


alphas = np.array(get_alphas(NNs, X_at))


plt.figure(figsize=(10,10))
for i in range(num_models):
    a = alphas[:,:,i]
    plt.subplot(5,3, i + 1)
    plt.plot(np.mean(recube(np.mean(a, axis=0)), axis=(1,2)))
    plt.title(df.columns[i][:10])
    plt.ylim([0,0.4])
    if i <12:
        plt.xticks([], [])
    if i % 3 != 0:
        plt.yticks([], [])
plt.savefig('Mean_Alphas.png')


plt.figure(figsize=(6,10))
for i in range(num_models):
    a = alphas[:,:,i]
    a_mean = np.mean(np.mean(recube(np.mean(a, axis=0)), axis=(1,2)).reshape(-1, 12), axis=0)
    # find std across ensemble
    ens_alphas = np.zeros([12 , alphas.shape[0]])
    for j in range(alphas.shape[0]):
        ens_alphas[:, j] = np.mean(np.mean(recube(a[j]), axis=(1,2)).reshape(-1, 12), axis=0)
    a_std = np.std(ens_alphas, axis=1)
    plt.subplot(5,3, i + 1)
    plt.fill_between(np.arange(12) + 1, a_mean - a_std, a_mean + a_std, alpha=0.2)
    plt.plot(np.arange(12) + 1, a_mean)
    plt.title(df.columns[i][:10])
    plt.ylim([0,0.4])
    if i <12:
        plt.xticks([], [])
    if i % 3 != 0:
        plt.yticks([], [])

plt.savefig('Seasonal_Alpha.png')

plt.figure(figsize=(10,10))
vmax = 0
vmin = 0
# for i in range(num_models):
#     if np.max(np.mean(recube(np.mean(np.array(alphas)[:,:,i], axis=0)), axis=0)) > vmax:
#         vmax = np.max(np.mean(recube(np.mean(np.array(alphas)[:,:,i], axis=0)), axis=0))

for i in range(num_models):
    plt.subplot(5,3, i + 1)
    plt.pcolormesh(np.mean(recube(np.mean(np.array(alphas)[:,:,i], axis=0)), axis=0))#, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(df.columns[i])

plt.savefig('Contour_Alpha.png')


# Model bias


betas = np.array(get_betas(NNs, X_at))
beta = np.mean(betas, axis=0)


plt.figure()
for b in betas:
    plt.plot(np.mean(recube(b.ravel()), axis=(1,2)))
plt.show()


plt.figure()
plt.pcolormesh(recube(beta)[0])
plt.colorbar()
plt.savefig('Contour_Beta.png')


# Noise predictions


aletoric_noise = []

for NN in NNs:
    feed_b = {}
    feed_b[NN.inputs] = X_at
    feed_b[NN.y_target] = y_at
    noise_sq = sess.run(NN.noise_sq, feed_dict=feed_b)
    aletoric_noise.append(noise_sq)


plt.figure()
plt.pcolormesh(np.mean(recube(np.sqrt(np.mean(np.array(aletoric_noise), axis=0))), axis=0))
plt.colorbar()
plt.savefig('Aleatoric.png')


err_pred = np.zeros((12,64))

for i in range(12):
    err_pred[i, :] = np.mean(recube(np.sqrt(np.mean(np.array(aletoric_noise), axis=0)))[i::12], axis=(0,2))
    
plt.figure()
plt.pcolormesh(err_pred.T)
plt.colorbar()
plt.savefig('Monthly_Aleratoric.png')


# Average noise in time


plt.figure()
plt.plot(np.arange(372), np.mean(recube(np.sqrt(np.mean(np.array(aletoric_noise), axis=0))), axis=(1,2)))

X_oos = X_at.copy()
X_oos[:, -1] = X_oos[:, -1] + 2

aletoric_noise = []

for NN in NNs:
    feed_b = {}
    feed_b[NN.inputs] = X_oos
    feed_b[NN.y_target] = y_at
    noise_sq = sess.run(NN.noise_sq, feed_dict=feed_b)
    aletoric_noise.append(noise_sq)

plt.plot(np.arange(372) + 372, np.mean(recube(np.sqrt(np.mean(np.array(aletoric_noise), axis=0))), axis=(1,2)))
plt.savefig('Noise_Time.png')


# What is the out of sample epistemic ucertainty


y_preds_at, y_preds_mu_at, y_preds_std_at,  y_preds_std_at_epi, y_preds_noisesq_at = fn_predict_ensemble(NNs,X_at)
y_preds_oos, y_preds_mu_oos, y_preds_std_oos,  y_preds_std_oos_epi, y_preds_noisesq_oos = fn_predict_ensemble(NNs,X_oos)




plt.figure()
plt.plot(np.arange(372), np.mean(recube(y_preds_std_at_epi), axis=(1,2)))

plt.plot(np.arange(372) + 372, np.mean(recube(y_preds_std_oos_epi), axis=(1,2)))
plt.savefig('OOS_Epi.png')


# What about overall uncertainty


plt.figure()
plt.plot(np.arange(372), np.mean(recube(y_preds_std_at), axis=(1,2)))
plt.plot(np.arange(372) + 372, np.mean(recube(y_preds_std_oos), axis=(1,2)))
plt.savefig('Overall_OOS.png')




