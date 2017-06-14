import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 15-46
lb = [7, 6, 4, 4, 5,\
		5, 5, 5, 5, 5,\
		6, 6, 6, 5, 2,\
		4, 3, 4, 4, 4,\
		3, 5, 5, 3, 3,\
		5, 4, 5, 5, 4, 4]

# 46-57
lb2 = [3, 3, 5, 5, 5,\
		 4, 2, 3, 5, 5, 6]
m, n = (180, 240)

def loadDataSet():
	data = []
	for i in range(15, 46):
		pic = cv2.imread('%d.jpg' % i, 1)
		small = cv2.resize(pic, (n, m))
		data.append(small)
	labels = np.array([lb], dtype='float32').transpose()
	data = np.array(data, dtype='float32')
	data = np.reshape(data, (data.shape[0], m, n, 3))
	print(data.shape); print(labels.shape)
	return data, labels

def loadTestSet():
	data = []
	for i in range(46, 57):
		pic = cv2.imread('%d.jpg' % i, 1)
		small = cv2.resize(pic, (n, m))
		data.append(small)
	labels = np.array([lb2], dtype='float32').transpose()
	data = np.array(data, dtype='float32')
	data = np.reshape(data, (data.shape[0], m, n, 3))
	print(data.shape); print(labels.shape)
	return data, labels

def weight(shape, name = 'w'):
	initial = tf.truncated_normal(shape, stddev=0.1, name=name)
	return tf.Variable(initial)

def bias(shape, name = 'b'):
	initial = tf.constant(0.1, shape=shape, name=name)
	return tf.Variable(initial)

def conv2d(x, W, name = 'conv'):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1],
			padding='SAME', name=name)

def max_pool_2x2(x, name='pool_2x2'):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name=name)

def max_pool_3x3(x, name='pool_3x3'):
	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1],
			padding='SAME', name=name)

def trans2visiable(conv, iy, ix, cy, cx):
	v = tf.slice(conv, (0, 0, 0, 0),  (1, -1, -1, -1))
	v = tf.reshape(v, (iy, ix, cy*cx)) # 60*80*64
	v = tf.transpose(v, (0, 2, 1)) # 60 * 64 * 80
	v = tf.reshape(v, (iy, cy, cx, ix)) # 60 * 8 * 8 * 80
	v = tf.transpose(v, (1, 0, 2, 3)) # 8 * 60 * 8 * 80
	v = tf.reshape(v, (1, iy*cy, ix*cx, 1))
	return v

def train():
	g = tf.Graph()
	data, labels = loadDataSet()
	tst_dt, tst_lb = loadTestSet()
	with g.as_default():
		X = tf.placeholder(tf.float32, [None, m, n, 3])
		Y = tf.placeholder(tf.float32, [None, 1])
		img = tf.Variable(np.zeros((480, 640), dtype='int32'))

		W1 = weight([10, 10, 3, 64], 'w1')
		b1 = bias([64], 'b1')
		h_conv1 = conv2d(X, W1, 'conv1') + b1
		h_relu1 = tf.nn.relu(h_conv1)
		h_pool1 = max_pool_3x3(h_relu1) # 60*80
		loss1 = tf.nn.l2_loss(W1)
		img1 = trans2visiable(h_pool1, 60, 80, 8, 8)
		tf.summary.image('pool1', img1)

		W2 = weight([8, 8, 64, 64], 'w2')
		b2 = bias([64], 'b2')
		h_conv2 = conv2d(h_pool1, W2, 'conv2') + b2
		h_relu2 = tf.nn.relu(h_conv2)
		h_pool2 = max_pool_2x2(h_relu2) #30*40
		img2 = trans2visiable(h_pool2, 30, 40, 8, 8)
		loss2 = tf.nn.l2_loss(W2)
		tf.summary.image('pool2', img2)

		W3 = weight([5, 5, 64, 64], 'w3')
		b3 = bias([64], 'b3')
		h_conv3 = conv2d(h_pool2, W3, 'conv3') + b3
		h_relu3 = tf.nn.relu(h_conv3)
		h_pool3 = max_pool_2x2(h_relu3) # 15*20
		loss3 = tf.nn.l2_loss(W3)
		img3= trans2visiable(h_pool3, 15, 20, 8, 8)
		tf.summary.image('pool3', img3)

		# m/3/2/2 = m/12
		m1, n1 = (int(m/12), int(n/12))
		w_fc1 = weight([m1*n1*64, 1024])
		b_fc1 = bias([1024])
		h_pool3_flat = tf.reshape(h_pool3, [-1, m1*n1*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)
		loss4 = tf.nn.l2_loss(w_fc1)

		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		w_fc2 = weight([1024, 1])
		b_fc2 = bias([1])
		logits = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
		y_conv = logits
		loss5 = tf.nn.l2_loss(w_fc2)

		loss = tf.reduce_mean((Y - y_conv)**2)
		loss += (loss1 + loss2 + loss3 + loss4 + loss5) * 1e-4
		tf.summary.scalar('loss', loss)
		train_step = tf.train.AdamOptimizer(3e-3).minimize(loss)
		
		NUM_THREADS = 2
		sess = tf.Session(config=tf.ConfigProto(
    			intra_op_parallelism_threads=NUM_THREADS))
		saver = tf.train.Saver()
		if (os.path.exists('model/checkpoint')):
			saver.restore(sess, 'model/model.ckpt')
		else:
			sess.run(tf.global_variables_initializer())

		summary_op = tf.summary.merge_all()
		writer1 = tf.summary.FileWriter('log/train',sess.graph)
		writer2 = tf.summary.FileWriter('log/test',sess.graph)

		for i in range(1, 101):
			sess.run(train_step, feed_dict = {
				X : data, Y : labels, keep_prob : 0.5})
			summary = sess.run(summary_op, 
				feed_dict={X: data[0:1], Y: labels[0:1], keep_prob: 1.0})
			writer1.add_summary(summary, i)
			summary = sess.run(summary_op,
				{ X: tst_dt[0:1], Y: tst_lb[0:1], keep_prob: 1.0})
			writer2.add_summary(summary, i)

			if (i % 10 == 0):
				lloss = sess.run(loss, feed_dict = {
					X : tst_dt, Y : tst_lb, keep_prob : 1.0})
				print('step %d: loss=%f' % (i, lloss))

		path = saver.save(sess, 'model/model.ckpt')
		print('Model saved in file: %s' % path)

		TEST = True
		if (TEST):
			tst_dt, tst_lb = loadTestSet()
			y = sess.run(y_conv, feed_dict = {
				X : tst_dt, keep_prob :1.0})
			y += 0.5
			y_int = y.astype('int32')
			print(y_int)
			print(tst_lb)
