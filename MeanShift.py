import math
import numpy as np
import random
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

"""
Sample Points is an array of vectors. 
Each point of the vector is [x1,x2,x3...xn], where n is the number of dimensions
Domain is an array of array of point, each array giving points in that dimension

WINDOW MUST BE COMPUTED ACCORDING TO THE KERNEL
"""
class MeanShift(object):

	def computeGaussianKernelValue(self,x):
		kernel_value = math.pow((2*math.pi),-self.dimension/2)*math.exp(-1/2*x)
		return kernel_value

	"""
		NEED FURTHER IMPROVEMENT
		Might include the possibility of including more kernels in the future
		self.domain is of form [[x1,x2,...,xn],[y1,y2,...,yn],...]
	"""
	def gaussianKernelDensityEstimation(self):
		vectorFunc = np.vectorize(self.computeGaussianKernelValue)		
		self.range = []
		y = np.zeros(self.mesh[0].shape)
		# Going through all X_i's (sample points)
		for idx,val in enumerate(self.samplePoints):
			xx = np.zeros(self.mesh[0].shape)
			for i in range(0,self.dimension):
				xx += abs(self.mesh[i]-val[i])**2
			y += vectorFunc(xx)/self.bandwidth
		self.range = y

	"""
	point is a numpy vector of [x1,x2,x3,...,xn], where n is the number of dimensions
	radius is a scalar that defines the radius of the window
	"""
	def findWindow(self,point):
		window = []
		for sample_point_vec in self.samplePoints:
			# print sample_point_vec, point
			if np.linalg.norm(sample_point_vec-point) < self.window_size:
				window.append(sample_point_vec)
		return window

	"""
	'current_iteration' is a scalar
	'y_i' is an numpy vector of the form [x1,x2,...,xn], where n is the number of dimensions
	'window' is a subset of 'samplePoints'
	"""
	# POSSIBLE INCLUSION OF WINDOW
	def gradientAscent(self,y_i,window=[],current_iteration=0):
		if current_iteration == 0:
			window = self.findWindow(y_i)
		if current_iteration >= self.max_iterations:
			return y_i
		else:
			# computing numerator
			numerator = np.zeros(self.dimension)
			for sample_point_vec in window:
				weight = math.exp(math.pow(np.linalg.norm((y_i
					-sample_point_vec)/self.bandwidth),2))
				numerator += sample_point_vec * weight
			# computing denominator
			denominator = np.zeros(self.dimension)
			for sample_point_vec in window:
				weight = math.exp(math.pow(np.linalg.norm((y_i
					-sample_point_vec)/self.bandwidth),2))
				denominator += weight
			if denominator.all() == 0:
				return y_i
			y_i2 = numerator/denominator
			# difference from previous point
			# NOT SURE IF NORM IS THE BEST WAY
			if np.linalg.norm(y_i2 - y_i) < self.mean_shift_threshold:
				return y_i2
		w = self.findWindow(y_i2)
		return self.gradientAscent(y_i2,w,current_iteration+1)

	def MeanShift(self):
		allpts = self.generatePoints(self.domain)
		for pt in allpts:
			peak = self.gradientAscent(pt)
			peak = tuple([round(eachpt,2) for eachpt in peak.tolist()])
			pt_to_tuple = tuple(pt.tolist())
			# print peak, pt_to_tuple
			if not peak in self.clusters:
				self.clusters[peak] = []	
			self.clusters[peak].append(pt_to_tuple)
		return self.clusters

		# length of all arrays in mesh are same
		# for each_point_idx in xrange(0,len(self.mesh[0])):
		# 	pt = np.array([])
		# 	for dim in xrange(0,len(self.mesh)):
		# 		pt = np.append(pt,self.mesh[dim][each_point_idx])
		# 		peak = self.gradientAscent(pt)
		# 		peak = tuple([round(eachpt,2) for eachpt in peak.tolist()])
		# 		pt_to_tuple = tuple(pt.tolist())
		# 		# print peak, pt_to_tuple
		# 		if not peak in self.clusters:
		# 			self.clusters[peak] = []	
		# 		self.clusters[peak].append(pt_to_tuple)
		# return self.clusters

	"""
	Domain is an array of array of points in one dimension
	Generates all the possible points from the domain
	"""
	def generatePoints(self,domain):
		if domain == []:
			return [[]]
		x = domain.pop(0)
		explode_lst = self.generatePoints(domain)
		lst = []
		for each_element in x:
			for arr in explode_lst:
				vec = np.append(each_element,arr)
				lst.append(vec)
		return lst


	def generateRandomSamples(self):
		self.samplePoints = []
		for times in xrange(0,self.count):
			random_value = np.array([])
			for dim in range(0,self.dimension):
				random_value = np.append(random_value,random.choice(self.domain[dim]))
			self.samplePoints.append(random_value)

	def visualize2d(self):
		plt.plot(self.mesh[0],self.range)
		plt.plot(self.samplePoints,[0]*len(self.samplePoints),'ro')
		plt.show()

	def visualize3d(self):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		X = self.mesh[0]
		Y = self.mesh[1]
		Z = self.range
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
			linewidth=0, antialiased=False)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
		fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.show()

	def wireframe3d(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		X = self.mesh[0]
		Y = self.mesh[1]
		Z = self.range
		ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
		for val in self.samplePoints:
			ax.scatter(val[0],val[1],0.0)
			# ax.plot([val[0]]*10,[val[1]]*10,np.linspace(0,0.25,10))
		plt.show()

	def printAllValues(self):
		print "Printing Sampled Points"
		print self.samplePoints
		print "---------------------"
		print "Printing the Domain"
		print self.domain
		print "---------------------"
		print "Printing the range"
		print self.range
		print "---------------------"

	def __init__ (self,bandwidth,dimension,count,window_size,max_iterations,threshold):

		# temporary
		self.limit = 10

		self.max_iterations = max_iterations
		self.mean_shift_threshold = threshold
		self.window_size = window_size

		self.count = count
		self.bandwidth = bandwidth
		self.dimension = dimension
		self.domain = [np.arange(1,self.limit,0.1)]*self.dimension

		self.clusters = {}

		# Mesh created for later use
		if self.dimension > 1:
			self.mesh = np.meshgrid(*self.domain)
		else:
			self.mesh = np.array(self.domain)

		# Generate random sample points
		self.generateRandomSamples()

		############ TESTING ###########################
		# self.gaussianKernelDensityEstimation()

		self.clusters = self.MeanShift()
		for key in self.clusters:
			if len(self.clusters[key]) > 10:
				print key

		# print self.generatePoints([[1,2,3,4]])

		############ PLOTTING2D ########################
		# plt.plot(self.mesh[0],self.range)
		# plt.plot(self.samplePoints,[0]*len(self.samplePoints),'ro')

		# for each_peak in self.clusters:
		# 	plt.plot(self.clusters[each_peak],[0.01]*len(self.clusters[each_peak]),'g')
		# 	plt.plot([each_peak],[0.005],'rx')

		# plt.show()