import numpy as np
import tensorflow
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

# Func->1
def discriminator(in_shape=(32, 32, 3), conditional_gan=bool, n_classes=None):
  if conditional_gan is False:
    print('As no conditional gan, n_classes are ignored\n')
    input=Input(in_shape)

    cnv1=Conv2D(128, (3, 3), strides=(2, 2), padding='same')(input) #16x16x128
    act1=LeakyReLU(alpha=0.2)(cnv1)

    cnv2=Conv2D(128, (3, 3), strides=(2, 2), padding='same')(act1) #8x8x128
    act2=LeakyReLU(alpha=0.2)(cnv2)

    flat=Flatten()(act2) #8192
    drop=Dropout(0.4)(flat)
    out=Dense(1, activation='sigmoid')(drop)
    model=Model(input, out)

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    return model
  
  else:
    if n_classes is not None:
      in_label=Input(shape=(1,)) #shape->(1,)
      li=Embedding(n_classes, 50)(in_label) #shape->(1, 50)
      li=Dense(in_shape[0] * in_shape[1])(li) #shape->(1, 1024)
      li=Reshape((in_shape[0],in_shape[1], 1))(li) #32x32x1
      
      input=Input(in_shape) #32x32x3
      merge=Concatenate()([input, li]) #32x32x4 (3 for input shape 32x32x3 and 1 for the label 32x32x1)

      cnv1=Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge) #16x16x128
      act1=LeakyReLU(alpha=0.2)(cnv1)

      cnv2=Conv2D(128, (3, 3), strides=(2, 2), padding='same')(act1) #8x8x128
      act2=LeakyReLU(alpha=0.2)(cnv2)

      flat=Flatten()(act2) #8192
      drop=Dropout(0.4)(flat)
      out=Dense(1, activation='sigmoid')(drop)
      model=Model([input, in_label], out)

      model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

      return model
    
    else:
      raise TypeError('Please specify the Number of Classes, n_classes')



#test_disr=discriminator(conditional_gan=True, n_classes=10)
#print(test_desr.summary())
# Func->2
def generator(latent_dim, conditional_gan=bool, n_classes=None):
  if conditional_gan is False:
    print('As no conditional gan, n_classes are ignored\n')
    input=Input(shape=(latent_dim, ))
    dense=Dense(128*8*8)(input) #8192 nodes
    act=LeakyReLU(alpha=0.2)(dense)
    reshape=Reshape((8, 8, 128))(act) #8x8x128

    ups1=Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(reshape) #16x16x128
    act1=LeakyReLU(alpha=0.2)(ups1)

    ups2=Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(act1) #32x32x128
    act2=LeakyReLU(alpha=0.2)(ups2)

    gen=Conv2D(3, (8, 8), activation='tanh', padding='same')(act2) #32x32x3
    model=Model(input, gen)

    return model
  else:
    if n_classes is not None:
      in_label=Input(shape=(1,)) #shape->(1,)
      li=Embedding(n_classes, 50)(in_label) #shape->(1, 50)
      li=Dense(8 * 8)(li) #shape->(1, 1024)
      li=Reshape((8, 8, 1))(li) #32x32x1

      input=Input(shape=(latent_dim, ))
      dense=Dense(128*8*8)(input) #8192 nodes
      act=LeakyReLU(alpha=0.2)(dense)
      in_img=Reshape((8, 8, 128))(act) #8x8x128

      merge=Concatenate()([in_img, li])

      ups1=Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge) #16x16x128
      act1=LeakyReLU(alpha=0.2)(ups1)

      ups2=Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(act1) #32x32x128
      act2=LeakyReLU(alpha=0.2)(ups2)

      gen=Conv2D(3, (8, 8), activation='tanh', padding='same')(act2) #32x32x3
      model=Model([input,in_label], gen)

      return model
    else:
      raise TypeError('Please specify the Number of Classes, n_classes')


#test_gen=generator(latent_dim=100, conditional_gan=True, n_classes=10)
#print(test_gen.summary())
# Func->3
def define_gan(gen, dis, conditional_gan=bool):
  if conditional_gan is False:
    dis.trainable=False
    g_input=gen.input
    g_output=gen.output
    gan_output=dis(g_output)
    model=Model(g_input, gan_output)
    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

    return model
  else:
    dis.trainable=False
    g_noise, g_label=gen.input
    g_output=gen.output
    gan_output=dis([g_output, g_label])
    model=Model([g_noise, g_label], gan_output)
    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

    return model

#test_gan=define_gan(test_gen, test_disr, conditional_gan=True)
#print(test_gan.summary())

from tensorflow.keras.datasets.cifar10 import load_data

# Func->4
def load_real_samples(conditional_gan=bool):
  if conditional_gan is False:
    (X_train, _), (_, _)=load_data()
    X=X_train.astype('float32')
    X=(X-127.5)/127.5

    return X
  else:
    (X_train, y_train), (_, _)=load_data()
    X=X_train.astype('float32')
    X=(X-127.5)/127.5

    return [X_train, y_train]

# Func->5
def generate_real_samples(data, batch_size, conditional_gan=bool):
  if conditional_gan is False:
    id=np.random.randint(0, data.shape[0], batch_size)
    X=data[id]
    y=np.ones((batch_size, 1))

    return X, y
  else:
    imgs, labels=data
    id=np.random.randint(0, imgs.shape[0], batch_size)
    X, labels=imgs[id], labels[id]
    y=np.ones((batch_size, 1))

    return [X, labels], y

# Func->6
def generate_latent_points(lat_dim,  n_samples, conditional_gan=bool, n_classes=10):
  if conditional_gan is False:
    x_input=np.random.randn(lat_dim*n_samples)
    x_input=x_input.reshape(n_samples, lat_dim)

    return x_input
  else:
      x_input=np.random.randn(lat_dim*n_samples)
      x_input=x_input.reshape(n_samples, lat_dim)
      labels=np.random.randint(0, n_classes, n_samples)

      return [x_input, labels]
    
# Func->7
def generate_fake_samples(generator, lat_dim, batch_size, conditional_gan=bool, n_classes=10):
  if conditional_gan is False:
    x_input=generate_latent_points(lat_dim=lat_dim, n_samples=batch_size, conditional_gan=conditional_gan, n_classes=n_classes)
    X=generator.predict(x_input)
    y=np.zeros((batch_size, 1))

    return X, y
  else:
      x_input, input_labels=generate_latent_points(lat_dim=lat_dim, n_samples=batch_size, conditional_gan=conditional_gan, n_classes=n_classes)
      X=generator.predict([x_input, input_labels])
      y=np.zeros((batch_size, 1))

      return [X, input_labels], y

# Func->8
def train(generator, discriminator, gan, data, latent_dim, n_epochs, batch_size, conditional_gan=bool, save_model_path=''):
  if conditional_gan is False:
    batch_per_epoch=int(data.shape[0]/batch_size)
    half_batch=int(batch_size/2)

    for i in range(n_epochs):
      for j in range(batch_per_epoch):
        X_real, y_real=generate_real_samples(data=data, batch_size=half_batch, conditional_gan=conditional_gan)
        discriminator_real_loss, discriminator_real_accuracy=discriminator.train_on_batch(X_real, y_real)
        X_fake, y_fake=generate_fake_samples(generator=generator, lat_dim=latent_dim, batch_size=half_batch, conditional_gan=conditional_gan)
        discriminator_fake_loss, discriminator_fake_accuracy=discriminator.train_on_batch(X_fake, y_fake)

        X_gan=generate_latent_points(lat_dim=latent_dim, n_samples=batch_size, conditional_gan=conditional_gan)
        y_gan=np.ones((batch_size, 1))

        gan_loss=gan.train_on_batch(X_gan, y_gan)

        print('Epoch>%d, Batch %d/%d, dl_r=%.3f, da_r=%.3f, dl_f=%.3f, da_f=%.3f, gl=%.3f' %
				  (i+1, j+1, batch_per_epoch, discriminator_real_loss, discriminator_real_accuracy, discriminator_fake_loss, discriminator_fake_accuracy, gan_loss))
    
    generator.save(save_model_path)
  
  else:
    batch_per_epoch=int(data[0].shape[0]/batch_size)
    half_batch=int(batch_size/2)

    for i in range(n_epochs):
      for j in range(batch_per_epoch):
        [X_real, real_labels], y_real=generate_real_samples(data=data, batch_size=half_batch, conditional_gan=conditional_gan)
        discriminator_real_loss, discriminator_real_accuracy=discriminator.train_on_batch([X_real, real_labels], y_real)
        [X_fake, fake_labels], y_fake=generate_fake_samples(generator=generator, lat_dim=latent_dim, batch_size=half_batch, conditional_gan=conditional_gan, n_classes=10)
        discriminator_fake_loss, discriminator_fake_accuracy=discriminator.train_on_batch([X_fake, fake_labels], y_fake)

        [X_gan, labels_gan]=generate_latent_points(lat_dim=latent_dim, n_samples=batch_size, conditional_gan=conditional_gan, n_classes=10)
        y_gan=np.ones((batch_size, 1))

        gan_loss=gan.train_on_batch([X_gan, labels_gan], y_gan)

        print('Epoch>%d, Batch %d/%d, dl_r=%.3f, da_r=%.3f, dl_f=%.3f, da_f=%.3f, gl=%.3f' %
				  (i+1, j+1, batch_per_epoch, discriminator_real_loss, discriminator_real_accuracy, discriminator_fake_loss, discriminator_fake_accuracy, gan_loss))
    
    generator.save(save_model_path)