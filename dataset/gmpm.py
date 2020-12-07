import tensorflow as tf
import gin
import h5py
import numpy as np
import os


AUTOTUNE = tf.data.experimental.AUTOTUNE

@gin.configurable

def dataset(batch_size=32, image_size=32, buffer_size=10000):
    
    def _process_image(image):
        #image = tf.image.resize(image[:, :, None], (image_size, image_size))
        #image = tf.reshape(image,[32,32,1])
        #image = tf.cast(image, tf.float32) * (2.0 / 511) - 1.0

        return image

    def load_h5_dataset(directory):
        print(" --------------------------------- ")
        print("Strat loading Datasat from H5DF files...")
        data = []

        for filename in os.listdir(directory):
            if filename.endswith(".h5"):
                with h5py.File(filename, "r") as f:
                    a_group_key = list(f.keys())[0]
                    # Get the data
                    temp = list(f[a_group_key])
                    data.append(temp[1:])
                continue
            else:
                continue

        data_flat = [item for sublist in data for item in sublist]
        data_flat = np.stack(data_flat, axis=0)
        precent_train_test_split = 0.7
        train = data_flat[:int(np.floor(precent_train_test_split * data_flat.shape[0])), :]
        test = data_flat[int(np.floor(precent_train_test_split * data_flat.shape[0])) + 1:, :]
        print(" --------------------------------- ")
        print("Finish loading Datasat from H5DF files...")

        return train, test
    
    def clusters_to_images(pathToCluster, samples): 
        
        clusters = np.load(args.color_cluster_path)
        samples = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [32, 32, 3]).astype(np.uint8) for s in samples]
        
        return samples
        
    
    directory = "./"

    train, test = load_h5_dataset(directory)
    
    pathToCluster = "/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy" # TODO : add path to cluster dir
    train = clusters_to_images(train,pathToCluster)
    test = clusters_to_images(test,pathToCluster)

    train = (
        tf.data.Dataset.from_tensor_slices(train)
            .shuffle(buffer_size)
            .map(_process_image, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
    )

    test = (
        tf.data.Dataset.from_tensor_slices(test)
            .shuffle(buffer_size)
            .map(_process_image, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
    )

    return train, test
