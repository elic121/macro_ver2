import numpy as np
import subprocess
import warnings
import base64
import time
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)

# try:
#     import cv2
# except:
#     subprocess.check_call([sys.executable, '-m', 'pip',
#                           'install', '--upgrade', 'opencv-python','-q'])
#     import cv2

# try:
#     import urllib.request
# except:
#     subprocess.check_call([sys.executable, '-m', 'pip',
#                           'install', '--upgrade', 'urllib.request','-q'])
#     import urllib.request

try:
    from tensorflow.keras import layers
    from tensorflow import keras
    import tensorflow as tf
except:
    subprocess.check_call([sys.executable, '-m', 'pip',
                          'install', '--upgrade', 'tensorflow','-q'])
    from tensorflow.keras import layers
    from tensorflow import keras
    import tensorflow as tf


class macro:

    def __init__(self) -> None:

        print()
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=['0', '6', '3', '9', '4', '2', '1', '8', '5', '7'],
            num_oov_indices=0, mask_token=None)

        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=['0', '6', '3', '9', '4', '2', '1', '8', '5', '7'],
            num_oov_indices=0, mask_token=None, invert=True)

    def load_model(self, modelName):

        self.modelName = modelName
        print("="*50)
        print(f"{self.modelName} loading...")

        try:
            self.model = keras.models.load_model(self.modelName)
        except:
            return f"{self.modelName} loading failed!!!"

        self.prediction_model = keras.models.Model(self.model.get_layer(name='image').input,
                                                   self.model.get_layer(name='dense2').output)
        print("="*50)

    def encode_single_sample(self, img, label):

        img_width, img_height = 150, 50

        img = tf.io.read_file(img)

        img = tf.io.decode_png(img, channels=1)

        img = tf.image.convert_image_dtype(img, tf.float32)

        img = tf.image.resize(img, [img_height, img_width])

        img = tf.transpose(img, perm=[1, 0, 2])

        label = self.char_to_num(
            tf.strings.unicode_split(label, input_encoding='UTF-8'))

        return {'image': img, 'label': label}

    def dataset(self, x, y):

        test_dataset = tf.data.Dataset.from_tensor_slices((x, y))

        test_dataset = (
            test_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(1)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        return test_dataset

    def decode_batch_predictions(self, pred):

        input_len = np.ones(pred.shape[0]) * pred.shape[1]

        results = keras.backend.ctc_decode(
            pred, input_length=input_len, greedy=True)[0][0][:, :4]

        return [tf.strings.reduce_join(self.num_to_char(results)).numpy().decode('utf-8')]

    def base64_to_image(self, base64):

        resp = urllib.request.urlopen(base64)

        image = np.asarray(bytearray(resp.read()), dtype='uint8')

        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        return image

    def img_info(self, base64_string):

        start = time.time()

        with open('./model/macro/tmp.png', 'wb') as f:
            f.write(base64.b64decode(base64_string))

        test_dataset = self.dataset(['./model/macro/tmp.png'], [''])

        for batch in test_dataset.take(1):
            pred_texts = self.decode_batch_predictions(
                self.prediction_model.predict(batch['image']))

        print(f'\t[{pred_texts[0]}]  [{time.time()-start:.3f} sec]')

        return pred_texts[0]
