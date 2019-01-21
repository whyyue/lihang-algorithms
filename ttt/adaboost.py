




class Sign(object):
	"""
	阈值分类器

	有两种方向，
		1) x<v y=1
		2) x>v y=1
		v 是阈值轴
	因为是针对已经二值化后的MNIST数据集，所以v的取值只有3个{0, 1, 2}
	"""

	def __init__(self, features, labels, w):
		self.X = features				# 训练数据特征
		self.Y = labels					# 训练数据的标签
		self.N = len(labels)			# 训练数据大小

		self.w = w						# 训练数据权值分布

		self.indexes = [0, 1, 2]		# 阈值轴可选范围

	def _train_less_than_(self):
		"""
		寻找(x<v y=1)情况下的最优v
		"""

		index = -1
		error_score = 1000000

		for i in self.indexes:
			score = 0
			for j in range(self.N):
				val = -1
				if self.X[j] < i:
					val = 1

				if val * self.Y[j] < 0:
					score += self.w[j]

			if score < error_score:
				index = i
				error_score = score


		return index, error_score

	def _train_more_than_(self):
		"""
		寻找(x>v y=1) 情况下的最优v
		"""

		index = -1
		error_score = 1000000

		for i in self.indexes:
			score = 0
			for j in range(self.N):
				val = 1
				if self.X[j] < i:
					val = -1

				if val * self.Y[j] < 0:
					score += self.w[j]

			if score < error_score:
				index = i
				error_score = score
		return index, error_score

	def  train(self):
		global sign_time_count
		time1 = time.time()
		less_index, less_score = self._train_less_than_()
		time2 = time.time()
		sign_time_count += time2-time1

		if less_score < more_score:
			self.is_less = True
			self.index = less_index
			return less_score

		else:
			self.is_less = False
			self.index = more_index
			return more_score

	def predict(self, feature):
		if self.is_less > 0:
			if feature < self.index:
				return 1.0
			else:
				return -1.0
		else:
			if feature < self.index:
				return -1.0
			else:
				return 1.0

class AdaBoost(object):
	def __init__(self):
		pass

	def _init_parameters_(self, features, labels):
		self.X = features					# 训练集特征
		self.Y = labels						# 训练集标签

		self.n = len(features[0])			# 特征维度
		self.N = len(features)				# 训练集大小
		self.M = 10							# 分类器数目

		self.w = [1.0/self.N] * self.N 		# 训练集的权值分布
		self.alpha = []						# 分类器系数 公式8.2
		self.classifier = []				# (维度，分类器)，针对当前维度的分类器

	def _w_(self, index, classifier, i):
		"""
		公式8.4不算Zm
		"""

		return self.w[i] * math.exp(-self.alpha[-1] * self.Y[i] * classifier.predict(self.X[i][index]))


	def _Z_(self, index, classifier):
		"""
		公式8.5
		"""
		Z = 0

		for i in range(self.N):
			Z += self._w_(index, classifier, i)

		return Z

	def train(self, features, labels):
		self._init_parameters_(features, labels)

		for times in range(self.M):
			logging.debug('iterater %d ' % times)



			best_classifier = (100000, None, None)			# (误差率, 针对的特征, 分类器)

		em = best_classifier[0]

		# 分析用，之后删除 开始
		print('em is %s, index is %d' % (str(em), best_classifier[1]))

		time2 = 



