import tensorflow as tf
import numpy as np

# Meta
INPUT='test.txt'

# Helper fnc : read the corpus and outputs the list of word-indices.
def textToIndexList(inputFileName, my_dict):
	content = open(inputFileName, 'r').read()
	output_list = content.split()
	return [my_dict[el] for el in output_list] 

# Helper fnc : makes RNN input data from the list of word-indices.
def indexListToBatch(indexList, startIndex, timeStep, batchSize, inputDim):
	output_x = []
	output_y = []
	listLen = len(indexList)
	for index in range(batchSize):
		if startIndex + timeStep + index > listLen - 1 : break
		temp_x = indexList[startIndex + index:startIndex + timeStep + index]
		temp_y = indexList[startIndex + timeStep + index]

		for temp_x_index in range(len(temp_x)):
			oneHot = np.zeros(inputDim)
			oneHot[temp_x[temp_x_index]] = 1
			temp_x[temp_x_index] = oneHot

		oneHot = np.zeros(inputDim)
		oneHot[temp_y] = 1
			
		output_x.append(temp_x)
		output_y.append(oneHot)

	return np.asarray(output_x), np.asarray(output_y)

# Helper fnc : converts word-index tensor to one-hot tensor
def oneHotEncoding(train_x, train_y, inputDim):
	output_x = []
	output_y = []
	for batch_index in range(len(train_x)):
		timestep_list = []
		for timestep_index in range(len(train_x[0])):
			temp_x = np.zeros(inputDim)
			temp_x[train_x[batch_index][timestep_index]] = 1 
			timestep_list.append(temp_x)

# Helper fnc : returns the dictionary and the size of it given a text corpus
def makeDictionary(inputFileName):

	my_dict = {}
	rev_dict = {}
	index = 0
	for line in open(inputFileName):
		splits = line.split()
		for el in splits :
			if el not in my_dict : 
				my_dict[el] = index
				rev_dict[index] = el
				index += 1

	return index, my_dict, rev_dict

# Weight initializer helper
def weightInitializer(name, shape):
	return tf.get_variable(name, shape, initializer=tf.contrib.keras.initializers.he_normal())	

# make dictionary
dict_size, my_dict, rev_dict = makeDictionary(INPUT)

# TF Hyperparameters
timesteps = 256
inputDim = dict_size
nLSTMHidden = 128
batchSize = 100
epoch = 10

X = tf.placeholder(tf.float32, shape=[None,timesteps,inputDim])
Y = tf.placeholder(tf.float32, shape=[None,inputDim])

LSTMOutW = weightInitializer("LSTMOutW", [nLSTMHidden, inputDim])
LSTMOutB = weightInitializer("LSTMOutB", [inputDim])

def LSTMmodel(X):
	# current shape of X : [batch, timesteps, inputDim]
	# but we want 'timesteps' tensors of shape : [batch, inputDim]
	X = tf.unstack(X, timesteps, 1)

	LSTM_cell = tf.contrib.rnn.BasicLSTMCell(nLSTMHidden)

	# predict
	outputs, states = tf.contrib.rnn.static_rnn(LSTM_cell, X, dtype=tf.float32)

	# there are n_input outputs but
	# we only want the last output
	return tf.matmul(outputs[-1], LSTMOutW) + LSTMOutB

pred = LSTMmodel(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	indexList = textToIndexList(INPUT,my_dict)
	if len(indexList) % batchSize : batchCnt = ((len(indexList)-timesteps)/batchSize) + 1
	else : batchCnt = ((len(indexList)-timesteps)/batchSize)
	print "batch count: %d " % batchCnt

	for epochIndex in range(epoch):
		for index in range(batchCnt):
			train_x, train_y = indexListToBatch(indexList, index * batchSize, timesteps, batchSize, inputDim)
			loss, _, oneHot = sess.run([cost,optimizer,pred], feed_dict={X:train_x, Y:train_y})
			print "[E: %d][B: %d] loss: %f" % (epochIndex, index, loss)

			# Uncomment the below to get the text output.
			#for outputIndex in range(len(train_x)):
			#	x_list = train_x[outputIndex]
			#	x_list = [rev_dict[np.argmax(el)] for el in x_list]
			#	print x_list[-5:], rev_dict[np.argmax(train_y[outputIndex])]
		
		
