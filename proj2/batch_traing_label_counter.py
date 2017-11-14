import re

Matrix = [[0 for x in range(10000)] for y in range(10000)] 

with open("generateBySentencesLog.txt", "r") as fp:
	previous_x = 0
	for line in fp:
		if 'batch_inputs' in line:
			previous_x = int((re.search('\d+', line)).group())
		if 'batch_labels' in line:
			label = int((re.search('\d+', line)).group())
			Matrix[previous_x][label] += 1

for i in range(10000):
	for j in range(10000):
		# print('i: %d, j: %d, occurrences: %d' % {i, j, Matrix[i][j]})
		if Matrix[i][j] > 30:
			print('i: ',i, 'j: ', j, 'Matrix[i][j]: ', Matrix[i][j])
