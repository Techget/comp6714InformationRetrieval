from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('dev_set/') if isfile(join('dev_set/', f))]
# print(onlyfiles)

for of in onlyfiles:
	words_file = open(join('dev_set/', of),'r')
	for i in words_file:
		print(i.strip())


