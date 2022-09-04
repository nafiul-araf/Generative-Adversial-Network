import numpy as np
import tensorflow
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.models import *

KERNEL_SIZE=(4, 4)
STRIDES=(2, 2)
PADDING='SAME'
INITIALIZER=RandomNormal(stddev=0.02)
OPTIMIZER=Adam(learning_rate=0.0002, beta_1=0.5)

def block(input_layer, n_filetrs, batchnorm=True):
  b=Conv2D(filters=n_filetrs, kernel_size=KERNEL_SIZE, strides=STRIDES, padding=PADDING, kernel_initializer=INITIALIZER)(input_layer)
  if batchnorm:
    b=BatchNormalization()(b, training=True)
  b=LeakyReLU(alpha=0.2)(b)

  return b

def discriminator(image_shape):
  in_src_image=Input(shape=image_shape)
  in_target_image=Input(shape=image_shape)
  merge=Concatenate()([in_src_image, in_target_image])

  d=block(input_layer=merge, n_filetrs=64, batchnorm=False)
  d=block(input_layer=d, n_filetrs=128)
  d=block(input_layer=d, n_filetrs=256)
  d=block(input_layer=d, n_filetrs=512)

  d=Conv2D(filters=1, kernel_size=KERNEL_SIZE, strides=(1, 1), padding=PADDING, kernel_initializer=INITIALIZER)(d)
  patch_out=Activation('sigmoid')(d)

  model=Model([in_src_image, in_target_image], patch_out)

  model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', loss_weights=[0.5])

  return model

def encoder(input_layer, n_filetrs, batchnorm=True):
  e=Conv2D(filters=n_filetrs, kernel_size=KERNEL_SIZE, strides=STRIDES, padding=PADDING, kernel_initializer=INITIALIZER)(input_layer)
  if batchnorm:
    e=BatchNormalization()(e, training=True)
  e=LeakyReLU(alpha=0.2)(e)

  return e

def decoder(input_layer, skip_connection_layer, n_filters, dropout=True):
  d=Conv2DTranspose(filters=n_filters, kernel_size=KERNEL_SIZE, strides=STRIDES, padding=PADDING, kernel_initializer=INITIALIZER)(input_layer)
  d=BatchNormalization()(d, training=True)
  if dropout:
    d=Dropout(0.5)(d, training=True)
  d=Concatenate()([d, skip_connection_layer])
  d=Activation('sigmoid')(d)

  return d

def generator(image_shape):
  input=Input(shape=image_shape)
  e1=encoder(input_layer=input, n_filetrs=64, batchnorm=False)
  e2=encoder(input_layer=e1, n_filetrs=128)
  e3=encoder(input_layer=e2, n_filetrs=256)
  e4=encoder(input_layer=e3, n_filetrs=512)
  e5=encoder(input_layer=e4, n_filetrs=512)
  e6=encoder(input_layer=e5, n_filetrs=512)
  e7=encoder(input_layer=e6, n_filetrs=512)

  bottleneck=Conv2D(filters=512, kernel_size=KERNEL_SIZE, strides=STRIDES, padding=PADDING, kernel_initializer=INITIALIZER)(e7)
  b=Activation('relu')(bottleneck)

  d1=decoder(input_layer=b, skip_connection_layer=e7, n_filters=512)
  d2=decoder(input_layer=d1, skip_connection_layer=e6, n_filters=512)
  d3=decoder(input_layer=d2, skip_connection_layer=e5, n_filters=512)
  d4=decoder(input_layer=d3, skip_connection_layer=e4, n_filters=512, dropout=False)
  d5=decoder(input_layer=d4, skip_connection_layer=e3, n_filters=256, dropout=False)
  d6=decoder(input_layer=d5, skip_connection_layer=e2, n_filters=128, dropout=False)
  d7=decoder(input_layer=d6, skip_connection_layer=e1, n_filters=64, dropout=False)

  g=Conv2DTranspose(filters=image_shape[2], kernel_size=KERNEL_SIZE, strides=STRIDES, padding=PADDING, kernel_initializer=INITIALIZER)(d7)
  output=Activation('tanh')(g)

  model=Model(input, output)
  return model

def gan_model(gen, dis, image_shape):
  for layer in dis.layers:
    if not isinstance(layer, BatchNormalization):
      dis.trainable=False
  in_img=Input(shape=image_shape)
  gen_out=gen(in_img)
  dis_out=dis([in_img, gen_out])
  model=Model(in_img, [dis_out, gen_out])

  model.compile(optimizer=OPTIMIZER, loss=['binary_crossentropy', 'mae'], loss_weights=[1, 100])
  return model

def generate_real_samples(data, n_samples, patch_size):
  train_a, train_b=data
  id=np.random.randint(0, train_a.shape[0], n_samples)
  X1, X2=train_a[id], train_b[id]
  y=np.ones((n_samples, patch_size, patch_size, 1))

  return [X1, X2], y

def generate_fake_samples(gen, n_samples, patch_size):
  X=gen.predict(n_samples)
  y=np.zeros((len(X), patch_size, patch_size, 1))

  return X, y

def train(gen, dis, gan, data, n_epochs, batch_size=1, save_path=''):
  n_patch=dis.output_shape[1]
  train_a, train_b=data
  batch_per_epoch=int(len(train_a)/batch_size)
  n_steps=batch_per_epoch*n_epochs

  for i in range(n_steps):
    [TRAIN_REAL_A, TRAIN_REAL_B], real_y=generate_real_samples(data, batch_size, n_patch)
    TRAIN_FAKE_B, fake_y=generate_fake_samples(gen, TRAIN_REAL_A, n_patch)
    dis_loss1=dis.train_on_batch([TRAIN_REAL_A, TRAIN_REAL_B], real_y)
    dis_loss2=dis.train_on_batch([TRAIN_REAL_A, TRAIN_FAKE_B], fake_y)
    gan_loss, _, _=gan.train_on_batch(TRAIN_REAL_A, [real_y, TRAIN_REAL_B])

    
    print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, dis_loss1, dis_loss2, gan_loss))
  
  gen.save(save_path)