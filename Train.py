import os, sys
import numpy as np
from matplotlib import pyplot as plt
from bpe import BPE
from scipy import stats

SEQ_SIZE = 24
TITLE_EMBED_SIZE = 36
TOKEN_EMBED_SIZE = 200
USE_GRU = True
USE_CATS = False
USE_AUTOENC = False
NUM_EPOCHS = 100
BATCH_SIZE = 200
LR = 0.001
DO_RATE = 0.5
BN = 0.99
SAVE_DIR = 'model_cats'
DATA_DIR = 'training_data'
NUM_RAND_GEN = 10

#Create directory to save model
if not os.path.exists(SAVE_DIR):
	os.makedirs(SAVE_DIR)

#Load bpe
print('Loading BPE...')
bpe = BPE()
bpe.load(DATA_DIR + '/words800.bpe')
end_token = bpe.str_to_token['\n']
bpe_size = len(bpe.str_to_token)
print('Loaded ' + str(bpe_size) + ' bpe tokens.')

#Load the categories
print('Loading Categories...')
all_categories = {}
with open(DATA_DIR + '/categories.txt', 'r') as fin:
	for line in fin:
		all_categories[line[:-1]] = len(all_categories)
num_categories = len(all_categories)
if USE_CATS:
	TITLE_EMBED_SIZE = num_categories
print('Loaded ' + str(num_categories) + ' categories')

#Create training samples
try:
	print('Loading Titles...')
	i_train = np.load(DATA_DIR + '/i_train.npy')
	c_train = np.load(DATA_DIR + '/c_train.npy')
	x_train = np.load(DATA_DIR + '/x_train.npy')
	x1_train = np.load(DATA_DIR + '/x1_train.npy')
	y_train = np.load(DATA_DIR + '/y_train.npy')
	if x_train.shape[1] != SEQ_SIZE:
		raise
except:
	print('Encoding Titles...')
	i_train = []
	c_train = []
	x_train = []
	x1_train = []
	y_train = []
	with open(DATA_DIR + '/titles_cats.txt', 'r') as fin:
		num_titles = 0
		for line in fin:
			title, category = line[:-1].lower().split('"')
			title = title + '\n'
			if category == '': category = 'other'
			c_vec = np.zeros((num_categories,), dtype=np.float32)
			c_vec[all_categories[category]] = 1.0
			encoded = np.array(bpe.encode(title), dtype=np.int32)
			seq_len = encoded.shape[0]
			first_len = min(SEQ_SIZE, seq_len) - 1
			x = np.full((SEQ_SIZE,), end_token)
			y = np.full((SEQ_SIZE,), end_token)
			x[1:1+first_len] = encoded[:first_len]
			y[:1+first_len] = encoded[:1+first_len]
			x1 = np.copy(x)
			i_train.append(num_titles)
			c_train.append(c_vec)
			x_train.append(x)
			x1_train.append(x1)
			y_train.append(y)
			if seq_len > SEQ_SIZE:
				for i in range(seq_len - SEQ_SIZE):
					x = encoded[i:i+SEQ_SIZE]
					y = encoded[i+1:i+SEQ_SIZE+1]
					i_train.append(num_titles)
					c_train.append(c_vec)
					x_train.append(x)
					x1_train.append(x1)
					y_train.append(y)
			num_titles += 1
			if num_titles % 1000 == 0:
				print('  ' + str(num_titles))
	i_train = np.array(i_train, dtype=np.int32)
	i_train = np.expand_dims(i_train, axis=1)
	c_train = np.array(c_train, dtype=np.int32)
	x_train = np.array(x_train, dtype=np.int32)
	x1_train = np.array(x1_train, dtype=np.int32)
	y_train = np.array(y_train, dtype=np.int32)
	np.save(DATA_DIR + '/i_train.npy', i_train)
	np.save(DATA_DIR + '/c_train.npy', c_train)
	np.save(DATA_DIR + '/x_train.npy', x_train)
	np.save(DATA_DIR + '/x1_train.npy', x1_train)
	np.save(DATA_DIR + '/y_train.npy', y_train)
assert(x_train.shape == y_train.shape)
assert(i_train.shape[0] == x_train.shape[0])
assert(i_train.shape[0] == x1_train.shape[0])
assert(i_train.shape[0] == c_train.shape[0])
assert(np.amax(x_train) < bpe_size)
assert(np.amax(y_train) < bpe_size)
num_titles = np.amax(i_train) + 1
num_samples = x_train.shape[0]
print("Loaded " + str(num_titles) + " titles.")
print("Loaded " + str(num_samples) + " training samples.")
y_train = np.expand_dims(y_train, axis=2)

#Load Keras and Theano
print("Loading Keras...")
import os, math
os.environ['KERAS_BACKEND'] = "tensorflow"
import tensorflow as tf
print("Tensorflow Version: " + tf.__version__)
import keras
print("Keras Version: " + keras.__version__)
from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, RepeatVector, TimeDistributed, LeakyReLU, CuDNNGRU, CuDNNLSTM, concatenate, SpatialDropout1D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Convolution1D
from keras.layers.embeddings import Embedding
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.models import Model, Sequential, load_model, model_from_json
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1
from keras.utils import plot_model, to_categorical
from keras import backend as K
K.set_image_data_format('channels_first')

#Fix bug with sparse_categorical_accuracy
def custom_sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())

def print_weight_shapes(model):
	for w in model.get_weights():
		print(w.shape)

def build_model(stateful):
	print("Building Model...")
	seq_size = (1 if stateful else SEQ_SIZE)
	bpe_embedding = Embedding(bpe_size, TOKEN_EMBED_SIZE, input_length=seq_size)

	if USE_AUTOENC:
		if stateful:
			ctxt_in = Input(batch_shape=(1,TITLE_EMBED_SIZE))
			ctxt_rep = RepeatVector(seq_size)(ctxt_in)
		else:
			ctxt_in = Input(shape=(seq_size,))
			ctxt_dense = bpe_embedding(ctxt_in)
			if USE_GRU:
				ctxt_dense = CuDNNGRU(TITLE_EMBED_SIZE, return_sequences=False, stateful=stateful, batch_size=1)(ctxt_dense)
			else:
				ctxt_dense = CuDNNLSTM(TITLE_EMBED_SIZE, return_sequences=False, stateful=stateful, batch_size=1)(ctxt_dense)
			ctxt_dense = BatchNormalization(momentum=BN)(ctxt_dense)
			ctxt_rep = RepeatVector(seq_size)(ctxt_dense)
			ctxt_rep = SpatialDropout1D(0.2)(ctxt_rep)
	elif USE_CATS:
		if stateful:
			ctxt_in = Input(batch_shape=(1,num_categories))
			ctxt_rep = RepeatVector(seq_size)(ctxt_in)
		else:
			ctxt_in = Input(shape=(num_categories,))
			ctxt_dense = GaussianDropout(0.2)(ctxt_in)
			ctxt_rep = RepeatVector(seq_size)(ctxt_dense)
	else:
		if stateful:
			ctxt_in = Input(batch_shape=(1,TITLE_EMBED_SIZE))
			ctxt_rep = RepeatVector(seq_size)(ctxt_in)
		else:
			ctxt_in = Input(shape=(1,))
			ctxt_dense = Embedding(num_titles, TITLE_EMBED_SIZE, input_length=1)(ctxt_in)
			ctxt_dense = Flatten(data_format='channels_last')(ctxt_dense)
			ctxt_rep = RepeatVector(seq_size)(ctxt_dense)
			ctxt_rep = SpatialDropout1D(DO_RATE)(ctxt_rep)

	if stateful:
		past_in = Input(batch_shape=(1,seq_size))
	else:
		past_in = Input(shape=(seq_size,))
	past_dense = bpe_embedding(past_in)

	x = concatenate([ctxt_rep, past_dense])
	x = Dropout(DO_RATE)(x)
	if USE_GRU:
		x = CuDNNGRU(360, return_sequences=True, stateful=stateful, batch_size=1)(x)
	else:
		x = CuDNNLSTM(360, return_sequences=True, stateful=stateful, batch_size=1)(x)

	x = TimeDistributed(BatchNormalization(momentum=BN))(x)
	x = TimeDistributed(Dense(bpe_size, activation='softmax'))(x)

	if stateful:
		return Model(inputs=[ctxt_in, past_in], outputs=[x])
	else:
		return Model(inputs=[ctxt_in, past_in], outputs=[x]), Model(ctxt_in, ctxt_dense)

#Build the training models
model, encoder = build_model(stateful=False)
model.compile(optimizer=Adam(lr=LR), loss='sparse_categorical_crossentropy', metrics=[custom_sparse_categorical_accuracy])
model.summary()
with open(SAVE_DIR + '/model.txt', 'w') as fout:
	model.summary(print_fn=lambda x: fout.write(x + '\n'))
#plot_model(model, to_file=SAVE_DIR + '/model.png', show_shapes=True)

#Also build a test model for testing
test_model = build_model(stateful=True)
first_layer_ix = len(model.get_weights()) - len(test_model.get_weights())

#Encoder Decoder
rand_vecs = np.random.normal(0.0, 1.0, (NUM_RAND_GEN, TITLE_EMBED_SIZE))

def calculate_pca():
	if USE_AUTOENC:
		x_enc = encoder.predict(x1_train, batch_size=BATCH_SIZE)
	elif USE_CATS:
		x_enc = encoder.predict(c_train, batch_size=BATCH_SIZE)
	else:
		x_enc = encoder.predict(i_train, batch_size=BATCH_SIZE)
	x_mean = np.mean(x_enc, axis=0)
	x_stds = np.std(x_enc, axis=0)
	x_cov = np.cov((x_enc - x_mean).T)
	u, s, x_evecs = np.linalg.svd(x_cov)
	x_evals = np.sqrt(s)
	print("Means: ", x_mean[:6])
	print("Evals: ", x_evals[:6])
	return x_mean, x_stds, x_evals, x_evecs

def save_pca(write_dir, pca):
	x_mean, x_stds, x_evals, x_evecs = pca
	np.save(write_dir + '/means.npy', x_mean)
	np.save(write_dir + '/stds.npy', x_stds)
	np.save(write_dir + '/evals.npy', x_evals)
	np.save(write_dir + '/evecs.npy', x_evecs)

	try:
		plt.clf()
		x_evals[::-1].sort()
		plt.title('evals')
		plt.bar(np.arange(x_evals.shape[0]), x_evals, align='center')
		plt.tight_layout()
		plt.draw()
		plt.savefig(write_dir + '/evals.png')

		plt.clf()
		plt.title('means')
		plt.bar(np.arange(x_mean.shape[0]), x_mean, align='center')
		plt.tight_layout()
		plt.draw()
		plt.savefig(write_dir + '/means.png')

		plt.clf()
		plt.title('stds')
		plt.bar(np.arange(x_stds.shape[0]), x_stds, align='center')
		plt.tight_layout()
		plt.draw()
		plt.savefig(write_dir + '/stds.png')
	except:
		pass

def encode_from_normal(pca, rand_vecs):
	x_mean, x_stds, x_evals, x_evecs = pca
	return x_mean + np.dot(rand_vecs * x_evals, x_evecs)

#Generation
def probs_to_ix(pk, is_first):
	pk *= pk
	pk /= np.sum(pk)
	xk = np.arange(pk.shape[0], dtype=np.int32)
	custm = stats.rv_discrete(name='custm', values=(xk, pk))
	return custm.rvs()

def generate(rand_vecs, pca, max_len):
	weights = model.get_weights()
	if USE_AUTOENC:
		weights = weights[:1] + weights[first_layer_ix+1:]
	else:
		weights = weights[first_layer_ix:]
	test_model.set_weights(weights)
	enc_vecs = encode_from_normal(pca, rand_vecs)
	for ix in range(rand_vecs.shape[0]):
		test_model.reset_states()
		i_sample = enc_vecs[ix:ix+1] #encoder.predict(np.array([[ix]], dtype=np.int32))
		x_sample = np.array([[end_token]], dtype=np.int32)
		all_samples = []
		for i in range(max_len):
			pred = test_model.predict([i_sample, x_sample])[0][0]
			y_sample = probs_to_ix(pred, i == 0)
			if y_sample == end_token:
				break
			all_samples.append(y_sample)
			x_sample = np.expand_dims(y_sample, 0)
		print(bpe.decode(all_samples))

#Utilites
def plotScores(scores, test_scores, fname, on_top=True):
	plt.clf()
	ax = plt.gca()
	ax.yaxis.tick_right()
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.grid(True)
	plt.plot(scores)
	plt.plot(test_scores)
	plt.xlabel('Epoch')
	plt.tight_layout()
	loc = ('upper right' if on_top else 'lower right')
	plt.draw()
	plt.savefig(fname)

#Train model
print("Training...")
train_loss = []
train_acc = []
test_loss = []
test_acc = []
all_ix = np.arange(num_samples)
batches_per_epoch = num_samples // BATCH_SIZE
for epoch in range(NUM_EPOCHS):
	if USE_AUTOENC:
		history = model.fit([x1_train, x_train], [y_train], batch_size=BATCH_SIZE, epochs=1)
	elif USE_CATS:
		history = model.fit([c_train, x_train], [y_train], batch_size=BATCH_SIZE, epochs=1)
	else:
		history = model.fit([i_train, x_train], [y_train], batch_size=BATCH_SIZE, epochs=1)
	loss = history.history['loss'][-1]
	acc = history.history['custom_sparse_categorical_accuracy'][-1]
	train_loss.append(loss)
	train_acc.append(acc)

	try:
		plotScores(train_loss, test_loss, SAVE_DIR + '/Loss.png', True)
		plotScores(train_acc, test_acc, SAVE_DIR + '/Acc.png', False)
	except:
		pass

	pca = calculate_pca()
	generate(rand_vecs, pca, 60)

	if loss == min(train_loss):
		model.save(SAVE_DIR + '/Model.h5')
		if not USE_CATS:
			save_pca(SAVE_DIR, pca)
		print("Saved")

	print("====  EPOCH FINISHED  ====")

print("Done")
