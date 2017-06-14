import cv2
import numpy as np

# retr_edge --> remove_small --> kmenas --> valid

def proc(filename):
	edges = retr_edge(filename)
	# remove_noise(edges)
	# cv2.imshow('edges', edges); k = cv2.waitKey()
	bowls = fill(edges, 300)
	# cv2.imshow('edges', bowls); k = cv2.waitKey()
	separated = bowls - edges
	separated[separated < 0] = 0
	separated = separated.astype('uint8')
	separated = hole_filling(separated)
	remove_noise(separated, 500)
	cv2.imshow('separated', separated); k = cv2.waitKey()
	# ctrs = kmeans(edges, 7)
	# valid(edges, ctrs)

def remove_noise(pic, thresh):
	ret, label, stats, centroids = cv2.connectedComponentsWithStats(pic)
	area = stats[:,cv2.CC_STAT_AREA]
	area[0] = 0
	dst = np.where(area < thresh)[0]
	m = dst.shape[0]
	for i in range(0, m):
		pic[label == dst[i]] = 0
	ret, label, stats, ctr = cv2.connectedComponentsWithStats(pic)
	n_merge = 0
	for i in range(1, ret - 1):
		for j in range(i + 1, ret):
			if np.sum((ctr[i]-ctr[j])**2) < 1600:
				ctr[i] = (ctr[i] + ctr[j]) / 2
				ctr[j] = ctr[i]
				n_merge += 1
	print('%d bowls detected' % (ret - 1 - n_merge))

def retr_edge(filename):
    src = cv2.imread(filename)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (240, 180))
    edges = cv2.Canny(small, 180, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.dilate(edges, kernel)
    # cv2.imshow('', edges); k = cv2.waitKey()
    return edges

def kmeans(pic, k):
	m, n = pic.shape[0:2]
	centroids = np.zeros((k, 2), dtype='int32')
	for i in range(0, k):
		centroids[i][0] = np.random.randint(0, m)
		centroids[i][1] = np.random.randint(0, n)

	pts = np.array(np.nonzero(pic), dtype='int32').transpose()
	n_pts = pts.shape[0]
	# randomly choose half of the points
	np.random.shuffle(pts)
	pts = pts[:int(n_pts / 4)]
	iters = 20

	for i in range(0, iters):
		cts_sum = np.zeros(centroids.shape, dtype='float64')
		count = np.zeros(k, dtype='int32')
		for p in pts:
			diff = centroids - p
			# the centroid point which is the closest one to p
			idx = np.argmin(diff[:,0] ** 2 + diff[:,1] ** 2)
			cts_sum[idx] += p * 0.001 # to avoid overflow
			count[idx] += 1
		# print(cts_sum)
		for i in range(0, k):
			if (count[i] != 0):
				centroids[i] = (cts_sum[i] / count[i] * 1000).astype('int32')

	pic1 = pic.copy()
	for i in range(0, k):
		cv2.circle(pic1, (centroids[i][1], centroids[i][0]),
			3, 127, 3)
	# cv2.imshow('', pic1); k = cv2.waitKey()
	# print(centroids)
	return centroids

# remove small block from edges
# to avoid effects on kmeans cluster
def fill(pic, thresh):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	pic = cv2.dilate(pic, kernel)
	pic = hole_filling(pic)
	ret, labels, stats, centroids = \
			cv2.connectedComponentsWithStats(pic)
	area = stats[:, cv2.CC_STAT_AREA]
	area[0] = 0
	new = np.zeros(pic.shape, dtype='uint8')
	for i in range(0, ret):
		if (area[i] > thresh):
			new[labels==i] = 255
	# cv2.imshow('', new); k = cv2.waitKey()
	new = cv2.erode(new, kernel)
	return new

def valid(pic, centroids):
	centroids = merge_close(centroids, 45)
	m, n = pic.shape
	i = 0
	# if there are white block near centroid, then keep the centroid
	# else remove it
	while (i < centroids.shape[0]):
		p = centroids[i]
		if (p[0] < 15 or p[0] > m - 15 or p[1] < 15 or p[1] > n - 15):
			centroids = np.delete(centroids, i, axis = 0)
		elif (np.count_nonzero(pic[p[0]-6:p[0]+6, p[1]-6:p[1]+6]) < 120):
			centroids = np.delete(centroids, i, axis=0)
		else:
			i += 1
	# print(centroids)
	for i in range(0, centroids.shape[0]):
		cv2.circle(pic, (centroids[i][1], centroids[i][0]),
			4, 127, 4)
	print('%d dishes detected!' % centroids.shape[0])
	cv2.imshow('', pic); k = cv2.waitKey()

def merge_close(centroids, thresh):
	m = centroids.shape[0]
	cp = centroids.copy(); new = []
	new.append(cp[0])
	cp = np.delete(cp, 0, axis=0)
	while (cp.shape[0] > 0):
		diff = np.array(new) - cp[0]
		diff = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
		idx = np.argmin(diff)
		if (diff[idx] < thresh):
			new[idx] = (new[idx] + cp[0]) / 2
		else:
			new.append(cp[0])
		cp = np.delete(cp, 0, axis=0)
		# print(len(new))
	return np.array(new).astype('int32')

def hole_filling(binary):
	binary[0,:] = 0; binary[-1,:] = 0; binary[:,0] = 0; binary[:,-1] = 0
	bin_inv = np.bitwise_not(binary)
	n, m = binary.shape
	ret, filled_inv1, mask, rect = cv2.floodFill(bin_inv, None, (0, 0), 0)
	res = (binary | filled_inv1)
	return res

