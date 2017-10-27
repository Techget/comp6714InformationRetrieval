import random
from scipy import spatial

matrix = []

for i in range(20):
	for j in range(5):
		matrix.append(random.sample(range(100), 5))


print(matrix[:2])

tree = spatial.KDTree(matrix)

test_list = random.sample(range(100), 5)
print(test_list)
print(tree.query(test_list))
print(matrix[tree.query(test_list)[1]])






