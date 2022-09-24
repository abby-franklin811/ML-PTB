# largely based on https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# and https://github.com/zhanxw/MB-GAN/blob/1972dcbc95a1ced429384bcb383b82700cb0cec8/model.py#L146

#originally formatted as a .ipynb, could be significantly more cleanly implemented

import pandas as pd
import numpy as np
import pickle


import keras.optimizers
from keras.optimizers import adam_v2 as A
import keras.backend as K

from keras.layers import Input, Dense, Dropout, Lambda, Layer
from keras.layers import BatchNormalization, Activation, LeakyReLU
from keras.layers.merge import _Merge
from keras.models import Sequential, Model

from functools import partial

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#############

#This model was trained twice (separately), once with preterm data and once with non-preterm data
#Which dataset should be used here?
# 0 means not preterm, 1 means preterm   (anything else will indescriminately use the whole dataset for training)
preterm = 0

#how many simulated samples to output into a .csv
N_TO_GENERATE = 5000

#############





#data read-in
##################################
phylotype_1 = pd.read_csv("phylotypes/phylotype_relabd.1e_1.csv")
metadata = pd.read_csv("metadata/metadata_normalized.csv")
answers = pd.read_csv("metadata/metadata.csv")


Ys = answers[['participant_id', 'was_preterm']].copy()

all_features = phylotype_1.merge(metadata, how='outer', on = 'specimen')

all_features = all_features.drop(columns=['collect_wk','Race: American Indian or Alaska Native', "Race: Asian", "Race: Black or African American", "Race: Native Hawaiian or Other Pacific Islander",
                           "Race: Unknown", "Race: White", "Ethnicity: Hispanic or Latino", "Ethnicity: Unknown"])

Y = Ys['was_preterm']
X = all_features.to_numpy()

X = phylotype_1.drop("specimen", axis=1).to_numpy()


if preterm == 0:
    X = phylotype_labeled.drop(phylotype_labeled[phylotype_labeled.was_preterm == 1].index).drop(columns=["specimen","was_preterm"], axis=1).to_numpy()
elif preterm == 1:
    X = phylotype_labeled.drop(phylotype_labeled[phylotype_labeled.was_preterm == 0].index).drop(columns=["specimen","was_preterm"], axis=1).to_numpy()
else:
    X = phylotype_1.drop("specimen", axis=1).to_numpy()

###############################################

#GAN code begins

###############################################

#GAN parameter definitions
###########################
BATCH_SIZE = 64

feature_size = X.shape[1]

latent_dim = 1000

LAMBDA = 10 # For gradient penalty

EPOCHS = 10000

N_CRITIC = 3 # Train critic(discriminator) n times then train generator 1 time.
N_GENERATOR = 1
LR = 1e-4

GRADIENT_PENALTY_WEIGHT = 10  # As per the paper





#losses and layers
##################

def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)



def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        batch_size = K.shape(inputs[0])[0]
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
    
    
#included to more accurately follow the MBGAN model
#without a tf_matrix (a matrix of phylo comparisons) doesn't do much
class PhyloTransform(Layer):
    def __init__(self, tf_matrix=None, **kwargs):
        if tf_matrix is None:
            self.kernel = None
        else:
            self.output_dim = tf_matrix.shape[1:]
            self.kernel = K.constant(tf_matrix, dtype='float32')
        super(PhyloTransform, self).__init__(**kwargs)

    def call(self, x):
        if self.kernel is None:
            return x
        else:
            return K.dot(x, self.kernel)
    
    def compute_output_shape(self, input_shape):
        if self.kernel is None:
            return input_shape
        else:
            return (input_shape[0], ) + self.output_dim
        

        
        
def make_generator(input_shape=latent_dim, output_units=feature_size, n_channels=512):
    model = Sequential()

    model.add(Dense(n_channels, activation="relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dense(n_channels))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dense(n_channels))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dense(output_units))
    model.add(Activation("softmax"))
    
    return model


def make_discriminator(input_shape=feature_size, n_channels=256, dropout_rate=0.25, tf_matrix=None, t_pow=1000.):
    model = Sequential()
    
    model.add(PhyloTransform(tf_matrix))
    #model.add(Dense(input_shape))
    model.add(Lambda(lambda x: K.log(1 + x * t_pow)/K.log(1 + t_pow)))
    model.add(Dense(n_channels))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_channels))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_channels))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    return model

















generator = make_generator()
discriminator = make_discriminator()

# Set discriminator to be not trainable in the generater as per https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=(latent_dim,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input],
                        outputs=[discriminator_layers_for_generator])
# These are the parameters used by https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py, which are recommended by Gulrajani et al.
# These could potentially be tuned
generator_model.compile(optimizer=A.Adam(0.0001, beta_1=0.5, beta_2=0.9),
                        loss=wasserstein_loss)

# After compiling, similar process for making generator non-trainable
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False


real_samples = Input(shape=X.shape[1:])
generator_input_for_discriminator = Input(shape=(latent_dim,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)


averaged_samples = RandomWeightedAverage()([real_samples,generated_samples_for_discriminator])


averaged_samples_out = discriminator(averaged_samples)


partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)

partial_gp_loss.__name__ = 'gradient_penalty'



discriminator_model = Model(inputs=[real_samples,
                                    generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])

discriminator_model.compile(optimizer=A.Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])





#training label vectors
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

for epoch in range(1, EPOCHS+1):
            for _ in range(N_CRITIC):
                # Randomly select a batch of samples to train the critic
                real = X[np.random.randint(0, X.shape[0], BATCH_SIZE)]
                noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
                d_loss = discriminator_model.train_on_batch([real, noise], [positive_y, negative_y, dummy_y])
            
            for _ in range(N_GENERATOR):
                noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
                g_loss = generator_model.train_on_batch(noise, positive_y)

            # Logging the progress
            log_info = [
                "iter={:d}".format(epoch), 
                "[D loss={:.6f}, w_loss_real={:.6f}, w_loss_fake={:.6f}, gp_loss={:.6f}]".format(*d_loss),
                "[G loss={:.6f}]".format(g_loss),
            ]
            print("{} {} {}".format(*log_info))



##############################################
#GAN code ends
##############################################

#generating from the GAN
#this is where to look to use the generated samples for other things
#or to save the trained GAN. But Keras doesn't play very nice with saving MBGAN's custom layers
#########################


X_labels = phylotype_1.drop("specimen", axis=1)


#use the concat step if the output should be real+fake data, instead of just fake data
if preterm == 0:
    real_X = phylotype_labeled.drop(phylotype_labeled[phylotype_labeled.was_preterm == 1].index).drop(columns=["specimen","was_preterm"], axis=1).to_numpy()
    fake_X = pd.DataFrame(gen(tf.random.normal([N_TO_GENERATE, latent_dim]), training=False).numpy(), columns = X_labels.columns)
    fake_X['was_preterm'] = 0
    fake_X.to_csv("GANedData_T.csv")
    #combo_X = pd.concat([real_X, fake_X], axis=0)
    #combo_X.to_csv("GANedData_ComboT.csv")

elif preterm == 1:
    real_X = phylotype_labeled.drop(phylotype_labeled[phylotype_labeled.was_preterm == 0].index).drop(columns=["specimen","was_preterm"], axis=1).to_numpy()
    fake_X = pd.DataFrame(gen(tf.random.normal([N_TO_GENERATE, latent_dim]), training=False).numpy(), columns = X_labels.columns)
    fake_X['was_preterm'] = 1
    fake_X.to_csv("GANedData_PT.csv")
    #combo_X = pd.concat([real_X, fake_X], axis=0)
    #combo_X.to_csv("GANedData_ComboPT.csv")
    
else:
    real_X = phylotype_1.drop("specimen", axis=1)
    fake_X = pd.DataFrame(gen(tf.random.normal([N_TO_GENERATE, latent_dim]), training=False).numpy(), columns = real_X.columns)
    fake_X.to_csv("GANedData.csv")
    #combo_X = pd.concat([real_X, fake_X], axis=0)
    #combo_X.to_csv("GANedData_Combo.csv")
    








#After training, the generated data can be appended to the real data and used to train downstream models



#optional quick TSNE plot to *roughly* verify that the simulated data makes sense
#from https://towardsdatascience.com/visualizing-high-dimensional-microbiome-data-eacf02526c3a
'''
from sklearn.manifold import TSNE

real_X['real'] = 1
fake_X['real'] = 0
combo_X = pd.concat([real_X, fake_X], axis=0)

tsne = TSNE(metric = 'jaccard', perplexity=30.0)
embeddings = tsne.fit_transform(combo_X.drop("real", axis=1))
tsne_df = pd.DataFrame(data = embeddings, columns = ['dim1', 'dim2'], index = combo_X["real"])
sns.scatterplot(x = 'dim1', y = 'dim2', hue = 'real', data = tsne_df)
'''

