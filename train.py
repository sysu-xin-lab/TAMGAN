import utils
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import model as md
from tqdm import tqdm
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    ### settings ###
    # model_path = "save_models/mymodel.h5"
    time = 46
    bands = 4
    batch_size = 8
    patch_size = 64
    epochs = 500
    lr = 0.001
    # lr_decay = 0.8
    os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BlUS_ID"
    os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BlUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    ### generate trainsets ###
    train_data = utils.readtraindata("./data/train/")
    train_data_aug = np.concatenate([train_data, np.rot90(train_data, k=1, axes=(2, 3)),
                                     np.rot90(train_data, k=2, axes=(2, 3)),
                                     np.rot90(train_data, k=3, axes=(2, 3))], axis=0)
    data, label = utils.generate_dataset(train_data_aug)
    del train_data, train_data_aug

    ### shuffle and make standard datasets ###
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.25, random_state=True)

    ### model defination / restore ###

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    # with strategy.scope():
    generator = md.generate_model(input_shape=(time, patch_size, patch_size, bands))
    discriminator = md.discriminate_model(input_shape=(time, patch_size, patch_size, bands))
    optimizer = keras.optimizers.Adam(lr)
    generator.compile(loss=md.loss_fun, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    gan = md.get_gan_network((time, patch_size, patch_size, bands), generator, discriminator, optimizer,
                             md.loss_fun)
    min_val_loss = 999

    ### strat training ###
    for e in range(epochs):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for i in tqdm(range(int(train_data.shape[0] / batch_size))):
            train_data_batch = train_data[i * batch_size:(i + 1) * batch_size]
            train_label_batch = train_label[i * batch_size:(i + 1) * batch_size]
            fake_image = generator.predict(train_data_batch)

            real_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            fake_Y = np.random.random_sample(batch_size) * 0.2

            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(train_label_batch[:, :, :, :, 0:4], real_Y)
            d_loss_fake = discriminator.train_on_batch(train_data_batch, fake_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            random_nums = np.random.randint(0, train_data.shape[0], size=batch_size)
            train_data_batch = train_data[random_nums]
            train_label_batch = train_label[random_nums]
            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(train_data_batch, [train_label_batch, gan_Y])
            print("discriminator_loss : %f" % discriminator_loss)
            print("gan_loss:", gan_loss)

        val_loss = 0
        for i in tqdm(range(int(test_data.shape[0] / batch_size))):
            val_data_batch = test_data[i * batch_size:(i + 1) * batch_size]
            val_label_batch = test_label[i * batch_size:(i + 1) * batch_size]
            val_loss += generator.evaluate(val_data_batch, val_label_batch)
        val_loss /= int(test_data.shape[0] / batch_size)

        ### save best model ###
        if (val_loss < min_val_loss):
            generator.save(f'./save_models/model_g.h5')
            discriminator.save(f'./save_models/model_d.h5')
            gan.save(f'./save_models/model.h5')
            min_val_loss = val_loss
        print("val_loss = ", val_loss, ",min_val_loss = ", min_val_loss)
