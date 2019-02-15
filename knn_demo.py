import math
import operator
import sys
import re


f = sys.argv[1]
file = open(f,'r');

D = []
#	Used regex to extract numbers and gender class from data file
for line in file:
	s = re.findall(r'-?\d+\.?\d*',line)
	s = list(map(int, s))
	l = re.findall(r'[a-z]|[A-Z](?=\))',line)
	D.append([s,l])
print("D =",D)
# D = [ ((170, 57, 32), 'W'),((192, 95, 28), 'M'),((150, 45, 30), 'W'),((170, 65, 29), 'M'),((175, 78, 35), 'M'),((185, 90, 32), 'M'),((170, 65, 28), 'W'),((155, 48, 31), 'W'),((160, 55, 30), 'W'),((182, 80, 30), 'M'),((175, 69, 28), 'W'),((180, 80, 27), 'M'),((160, 50, 31), 'W'),((175, 72, 30), 'M')]

file.close()
pointflag = int(input("\nEnter 1 for default test data or else enter 2 for entering test data manually :\n"))
if pointflag == 2:
	# p_input = input("Enter the number seperated by ','")
	# points = re.findall(r'-?\d+\.?\d*',p_input)
	# points = [tuple(map(int, points))]
	h = int(input("Enter Height :"))
	w = int(input("Enter weight :"))
	a = int(input("Enter age :"))
	points = [(h,w,a)]
else:points = [(155, 40, 35),(170, 70, 32),(175, 70, 35),(180, 90, 20)]
print(points)
distances = [[[] for i in range(len(D))] for j in range(len(points))]

while(True):
	print("\nTo calculate using KNN--> Enter 1")
	print("To calculate using Gaussian Naive Bayes, Enter 2\n")
	method_flag = int(input("Input: "))
	if method_flag not in [1, 2]:
		print("Error. Please enter either 1 or 2 to use KNN or Gaussian Naive Bayes\n")
		continue
	print("To calculate with age--> Enter 1")
	print("Else, Enter 2\n")
	age_flag = int(input("Input: "))
	if age_flag not in [1,2]:
		print("Error. Please enter either 1 or 2 to calculate with or without age")
	else:break

#	KNN classification
if method_flag == 1:
	while(True):
		k = int(input("Enter the Value of K :\t"))
		if(k%2 == 0):
			print("Error, Please enter a odd number")
		else:break

	#	calculating cartesian distance for all test data with all training data
	count1 = 0
	for point in points:
		count2 = 0
		for datarow in D:
			x_sqr_diff = math.pow(datarow[0][0] - point[0], 2)
			y_sqr_diff = math.pow(datarow[0][1] - point[1], 2)
			z_sqr_diff = math.pow(datarow[0][2] - point[2], 2)
			if age_flag == 1:
				cart_distance = math.sqrt(x_sqr_diff + y_sqr_diff + z_sqr_diff)
			elif age_flag == 2:
				cart_distance = math.sqrt(x_sqr_diff + y_sqr_diff)
			distances[count1][count2] = datarow,cart_distance
			print("Cartesian distances w.r.t point:",point)
			print(distances[count1][count2])
			count2 += 1;
		print("\n")
		count1 += 1;

	#	finding the K nearest Neighbour for all test points
	count1 = 0
	k_nearest = [[[] for i in range(k)] for j in range(len(points))]
	for i in distances:
		count2 = 0
		i.sort(key = operator.itemgetter(1));
		# print(i)
		for j in range(k):
			k_nearest[count1][count2] = i[j][0]
			count2 += 1
		count1 += 1
	print('#########	K-nearest		###########')

	#	Predicting the output based on the max of class of the k nearest points
	for kpoints in k_nearest:
		all_out_ans = []
		count_of_ans = []
		for point in kpoints:
			if point[1] not in all_out_ans:
				all_out_ans.append(point[1])
				count_of_ans.append(0)
			index = all_out_ans.index(point[1])
			count_of_ans[index] += 1
		print("\nClass of ",k, "nearest point and their count:")
		print("Classes: ",all_out_ans," and their count: ",count_of_ans)
		max_index = count_of_ans.index(max(count_of_ans))
		print('Predicted output for the point is:',all_out_ans[max_index])

#	Gaussian Naive Bayes classification
if method_flag == 2:
	classes = []
	class_prob = []
	if age_flag == 1:features = 3;
	elif age_flag == 2:features = 2

	#	calculate probability of classes
	for datarow in D:
		if datarow[1] not in classes:
			classes.append(datarow[1])
			class_prob.append(1)
		else:class_prob[classes.index(datarow[1])] += 1
	for index,c in enumerate(class_prob):
		class_prob[index] /= float(len(D))
	print("Classes :",classes)
	print("class_prob :", class_prob)

	mean_list = [0]*(len(classes)*features)
	var_list = [0]*(len(classes)*features)
	stdev_list = [0]*(len(classes)*features)
	gaussian_list = [0]*(len(classes)*features)
	#	calculate mean for each feature of each class
	for count,c in enumerate(classes):
		count *= features;
		nc = 0
		for datarow in D:
			if datarow[1] == c:
				nc += 1
				for feature in range(features):
					mean_list[count+feature] += datarow[0][feature]
		for feature in range(features):
			mean_list[count + feature] /= nc
	print("mean :\n",mean_list)

	#	calculate variance and stddev. for each feature of each class
	for count,c in enumerate(classes):
		count *= features;
		nc = 0
		for datarow in D:
			if datarow[1] == c:
				nc += 1
				for feature in range(features):
					var_list[count+feature] += math.pow((datarow[0][feature] - mean_list[count+feature]),2)
		for feature in range(features):
			var_list[count + feature] /= (nc - 1)
			stdev_list[count + feature] = math.sqrt(var_list[count + feature])

	#	classifying for all the test points
	for point in points:
		#	calulating normal distribution for features when class is given
		multi = []
		for count, c in enumerate(classes):
			count1 = count
			count *= features;
			temp = 1
			for feature in range(features):
				expo_ans = math.exp(-math.pow((point[feature] - mean_list[count + feature]),2)/(2*var_list[count + feature]))
				gaussian_list[count + feature] = expo_ans/math.sqrt(2*math.pi*var_list[count + feature])
				temp *= gaussian_list[count + feature]
			multi.append(class_prob[count1]*temp)

		print("\ngaussian_list for point: ",point," is :",gaussian_list)
		# print("Multiplciation :", multi)
		print("\n The test point",point," is classified as :", classes[multi.index(max(multi))])
