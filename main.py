import os

import time
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL

from config import *
from preprocessing import load_data_by_author, normalize
from generative_network import make_generator_model
from discriminative_network import make_discriminator_model

if os.path.exists(data_file) and not update_data_file:
    images = np.load(data_file)
    print(images.shape, "\n\n\n\n\n\n\n\n\n\n\n\n\n") # TODO: for some reason the shape is (128,) when it should be (160, 100, 72, 3)
else:
    images = load_data_by_author('Seb McKinnon', should_log=True)
    images = normalize(images)
    np.save(data_file, images)

dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = make_generator_model()
discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
    
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(normalize(predictions[i, :, :, 0], input_range=(-1, 1), output_range=(0, 255)), cmap='gray')
        plt.axis('off')

    plt.savefig('./images/epoch_{:04d}.png'.format(epoch))
    plt.close()

def train(dataset, epochs, progress_bar_width=80, progress_char='+', empty_char='-'):

    for epoch in range(epochs):
        start = time.time()

        gen_loss = -1
        disc_loss = -1

        cur_time = time.time()
        i = 0
        l = BUFFER_SIZE // BATCH_SIZE

        for image_batch in dataset:
            w = i * progress_bar_width // l
            s = int(time.time() - cur_time)
            print('[' + progress_char * w + empty_char * (progress_bar_width - w) + ']', i, '/', l, datetime.timedelta(seconds=s), end='\r')
            gen_loss, disc_loss = train_step(image_batch)
            i += 1
        
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % checkpoint_frequency == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start), ' ' * (progress_bar_width + 10))

        generate_and_save_images(generator, epochs, seed)

if checkpoint_restore:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(dataset, EPOCHS)

def display_image(epoch_no):
    return PIL.Image.open('./images/epoch_{:04d}.png'.format(epoch_no))
display_image(EPOCHS)

s = str(int(time.time() * 1000))
os.mkdir('./models/' + s)
os.mkdir('./models/' + s + '/generator')
os.mkdir('./models/' + s + '/discriminator')

generator.save('models/' + s + '/generator') 
discriminator.save('models/' + s + '/discriminator') 