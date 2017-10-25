import random
class InvertedList:
    def __init__(self, l):
        self.data = l[:] # make a copy
        self.cur = 0     # the cursor
 
    def get_list(self):
        return self.data
 
    def eol(self):
        # we use cur == len(list) to indicate EOL
        return False if self.cur < len(self.data) else True
 
    def next(self, val = 1):
        # does not allow cur to be out-of-range, but use len(list) to indicate EOL
        self.cur = min(self.cur + val, len(self.data))
 
    def elem(self):
        if self.eol():
            return None
        else:
            return self.data[self.cur]
    def peek(self, pos):
        # look at the element under the current cursor, but does not advance the cursor.
        if pos < len(self.data):
            return self.data[pos]
        else:
            return None
    def reset(self):
        self.cur = 0
import submission
 
def intersection_galloping(a, b):
    # just in case these lists have been traversed.
    a.reset()
    b.reset()
    count = 0
 
    ret = []
    while not a.eol() and not b.eol():
        if a.elem() == b.elem():
            ret.append(a.elem())
            a.next()  # Note that here you are only allowed to move the cursor of one InvertedList Object.
        else:
            if a.elem() < b.elem():
                count = count + submission.gallop_to(a,b.elem())
            else:
                count = count + submission.gallop_to(b,a.elem())
    # end_while
    return ret, count
 
a = InvertedList([2, 4, 6, 8, 10, 12, 14, 16, 18])
b = InvertedList([1, 2, 4, 8, 16, 32])
result = intersection_galloping(a, b)
print(result)
 
import time
t = (int)(time.time())
print ("random seed " + str(t))
random.seed(t)
print ("begin q1")
cc = 10
n = 100000
while cc > 0:
    cc -= 1
    # a = InvertedList( sorted(list(set(sorted([random.randint(1, 100) for _ in range(10)])))))
    # b = InvertedList( sorted(list(set(sorted([random.randint(1, 100) for _ in range(10)])))))
 
    a = InvertedList( sorted(list(set(([random.randint(1, 1000000) for _ in range(n)])))))
    b = InvertedList( sorted(list(set(([random.randint(1, 1000000) for _ in range(1000)])))))
    intersection = list(set(a.data).intersection(b.data))
    if (sorted(intersection) == intersection_galloping(a, b)[0]) == False:
 
        print ("different")
        # print (a.data)
        # print b.data)
        print (intersection)
        print (intersection_galloping(a, b)[0])
 
print ("end q1")
 
 
inputs = "1111111100000000011111111000001001010010111010"
result = submission.decode_gamma(inputs)
print(result)
inputs = "1110001000000001110001000010010100010011011011000111110010000"
result = submission.decode_delta(inputs)
print(result)
 
 
inputs = "1000101111100000"
result = submission.decode_gamma(inputs)
print(result)
 
inputs = "11000111001000100100000"
result = submission.decode_delta(inputs)
print(result)
 
 
 
inputs = "11110101111011"
b = 4
result = submission.decode_rice(inputs, b)
print(result)
 
 
import submission
 
random.seed(1999)
n = 13
 
print('~~~~~~~~~~~~~')

index = [random.randint(1, 100) for _ in range(n)]
index = [15, 46, 19, 93, 73, 64, 33, 80, 73, 26, 22, 77, 27]
result = submission.Logarithmic_merge(index, 10, 3) #cut_off = 10, initial buffer_size = 3
print(result) # expect to see a list of lists
result = submission.Logarithmic_merge(index, 10, 2) #cut_off = 10, initial buffer_size = 3
print(result) # expect to see a list of lists
 
 
print ('---------')
 
random.seed(1999)
n = 29
index = [random.randint(1, 100) for _ in range(n)]
 
result = submission.Logarithmic_merge(index, 25, 3) #cut_off = 10, initial buffer_size = 3
print(result) # expect to see a list of lists
print(index)
# n = 100000
# while 1:
#     index = [random.randint(1, 100) for _ in range(n)]
#     result = submission.Logarithmic_merge(index, 50000, 3) #cut_off = 10, initial buffer_size = 3
