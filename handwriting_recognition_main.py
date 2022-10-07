# import tools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

np.random.seed(42)
tf.random.set_seed(42)

# Get path for data and create array to store labels
base_path = r"C:\Users\samue\Documents\data"
words_list = []

words = open(f"{base_path}/words.txt", "r").readlines()
for line in words:
    if line[0] == "#":
        continue
    if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
        words_list.append(line)

np.random.shuffle(words_list)

# Split dataset  into 3 subsets: 90 percent for training, 5 percent for validation, and 5 percent for testing
split_idx = int(0.9 * len(words_list))
train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:]

val_split_idx = int(0.5 * len(test_samples))
validation_samples = test_samples[:val_split_idx]
test_samples = test_samples[val_split_idx:]

# verify that all the sets total up to the length of the original word list
assert len(words_list) == len(train_samples) + len(validation_samples) + len(
    test_samples
)

print(f"Total training samples: {len(train_samples)}")
print(f"Total validation samples: {len(validation_samples)}")
print(f"Total test samples: {len(test_samples)}")

base_image_path = os.path.join(base_path, "words")
base_test_image_path = os.path.join(base_path, "inputs")

def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples		


def get_input_image_paths():
	paths = []
	labels = []
	partI = "a01"
	partII = "a01-000u"
	partIII = "a01-000u-00-00"
	img_path = os.path.join(
		base_test_image_path, partI, partII, partIII + ".png"
	)
	paths.append(img_path)
	
	labels.append("Madrid")
	return paths, labels


train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
test_img_paths, test_labels = get_image_paths_and_labels(test_samples)
input_img_paths, input_labels = get_input_image_paths()

# Find maximum length and the size of the vocabulary in the training data.
train_labels_cleaned = []
characters = set()
max_len = 0

for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)

    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)

characters = sorted(list(characters))

print("Maximum length: ", max_len)
print("Vocab size: ", len(characters))

def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels


validation_labels_cleaned = clean_labels(validation_labels)
test_labels_cleaned = clean_labels(test_labels)

AUTOTUNE = tf.data.AUTOTUNE

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check the amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


batch_size = 64
padding_token = 99
image_width = 128
image_height = 32

def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)
test_ds = prepare_dataset(test_img_paths, test_labels_cleaned)
input_ds = prepare_dataset(input_img_paths, input_labels)


class CTCLayer(keras.layers.Layer):
	def __init__(self, name=None):
		super().__init__(name=name)
		self.loss_fn = keras.backend.ctc_batch_cost

	def call(self, y_true, y_pred):
		batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
		input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
		label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

		input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
		label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
		loss = self.loss_fn(y_true, y_pred, input_length, label_length)
		self.add_loss(loss)

		# At test time, just return the computed predictions
		return y_pred


def build_model():
	# Inputs to the model 128, 32
	input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
	labels = keras.layers.Input(name="label", shape=(None,))

	# First conv block
	x = keras.layers.Conv2D(
		32,
		(3, 3),
		activation="relu",
		kernel_initializer="he_normal",
		padding="same",
		name="Conv1",
	)(input_img)
	x = keras.layers.Conv2D(
		64,
		(3, 3),
		activation="relu",
		kernel_initializer="he_normal",
		padding="same",
		name="Conv2",
	)(x)

	x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

	# We have used two max pool with pool size and strides 2.
    	# Hence, downsampled feature maps are 4x smaller. The number of
    	# filters in the last layer is 64. Reshape accordingly before
    	# passing the output to the RNN part of the model.
	new_shape = ((image_width // 2), (image_height // 2) * 64)
	x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
	x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
	x = keras.layers.Dropout(0.2)(x)

	# RNNs
	x = keras.layers.Bidirectional(
		keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
	)(x)
	x = keras.layers.Bidirectional(
		keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
	)(x)
	
	# +2 is to account for the two special tokens introduced by the CTC loss.
	# The recommendation comes here: https://git.io/J0eXP.
	x = keras.layers.Dense(
		len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
	)(x)
	
	# Add CTC layer for calculating CTC loss at each step.
	output = CTCLayer(name="ctc_loss")(labels, x)

	# Define the model.
	model = keras.models.Model(
		inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
	)	
	# Optimizer
	opt = keras.optimizers.Adam()
	# Compile the model and return
	model.compile(optimizer=opt)
	return model

# Get the model.
model = build_model()
model.summary()

validation_images = []
validation_labels = []

for batch in validation_ds:
	validation_images.append(batch["image"])
	validation_labels.append(batch["label"])

def calculate_edit_distance(labels, predictions):
	# Get a single batch and convert its labels to sparse tensors.
	sparse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

	# Make predictions and convert them to sparse tensors.
	input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
	predictions_decoded = keras.backend.ctc_decode(
		predictions, input_length=input_len, greedy=True
	)[0][0][:, :max_len]
	sparse_predictions = tf.cast(
		tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
	)

	# Compute individual edit distances and average them out.
	edit_distances = tf.edit_distance(
		sparse_predictions, sparse_labels, normalize=False
	)
	return tf.reduce_mean(edit_distances)

class EditDistanceCallback(keras.callbacks.Callback):
	def __init__(self, pred_model):
		super().__init__()
		self.prediction_model = pred_model

	def on_epoch_end(self, epoch, logs=None):
		edit_distances = []
		
		for i in range(len(validation_images)):
			labels = validation_labels[i]
			predictions = self.prediction_model.predict(validation_images[i])
			edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

		
		print(
			f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
		)


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = build_model()

prediction_model = keras.models.Model(
	model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

edit_distance_callback = EditDistanceCallback(prediction_model)

# Check for previous training checkpoints and load weights if found
try:
	model.load_weights(checkpoint_path)

# Train the model if no previous training information
except:
	epochs = 50 # Epochs should be at least 50 for good results
	ds = train_ds
	exec(open("handwriting_recognition_training.py").read())

# A utility function to decode the output of the network.
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

#  Prediction method for input images.
def pred_test_data():
	for batch in test_ds.take(1):
		batch_images = batch["image"]
		_, ax = plt.subplots(4, 4, figsize=(15, 8))
		
		preds = prediction_model.predict(batch_images)
		pred_texts = decode_batch_predictions(preds)
		
		for i in range(16):
			img = batch_images[i]
			img = tf.image.flip_left_right(img)
			img = tf.transpose(img, perm=[1, 0, 2])
			img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
			img = img[:, :, 0]

			title = f"Prediction: {pred_texts[i]}"
			ax[i // 4, i % 4].imshow(img, cmap="gray")
			ax[i // 4, i % 4].set_title(title)
			ax[i // 4, i % 4].axis("off")
	plt.show()


def pred_input():
	for batch in input_ds.take(1):
		batch_images = batch["image"]
		_, ax = plt.subplots(1, 1, figsize=(10, 3))

		preds = prediction_model.predict(batch_images)
		pred_texts = decode_batch_predictions(preds)

		img = batch_images[0]
		img = tf.image.flip_left_right(img)
		img = tf.transpose(img, perm=[1, 0, 2])
		img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
		img = img[:, :, 0]

		title = f"Prediction: {pred_texts[0]}"
		ax.imshow(img, cmap="gray")
		ax.set_title(title)
		ax.axis("off")
	plt.show()

def train_input():
	epochs = 1
	ds = input_ds
	exec(open("handwriting_recognition_training.py").read())