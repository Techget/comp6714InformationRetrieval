## Submission.py for COMP6714-Project1

###################################################################################################################
## Question No. 0:
def add(a, b): # do not change the heading of the function
    return a + b




###################################################################################################################
## Question No. 1:

def gallop_to(a, val):# do not change the heading of the function
    #pass # **replace** this line with your code
    count = 0
    inc = 1
    
    ### first step: gallop ###
    gallop_val = a.peek()
    # the case where it starts with large value
    if gallop_val >= val:
        return count
    
    while gallop_val < val:
        gallop_val = a.peek()
        # it comes to an end. but still smaller than val, return directly, the wrap function will deal with this
        if gallop_val == None:
            return count
        
        if gallop_val < val:
            a.next(val = inc)
            inc *= 2
            count += 1
        else:
            count += 1
            break
    
    ### if the current val is equal to val, then we do not need to BS
    if a.peek() == val:
        return count

    ### binary search ###
    cur_cur = a.cur
    prev_cur = cur_cur - inc
    
    # do not need to make a copy, since no modification will be made
    data = a.data
    mid = (cur_cur + prev_cur) / 2
    
    while cur_cur - prev_cur > 1:
        if data[mid] < val:
            prev_cur = mid
            mid = (cur_cur + prev_cur) / 2
        elif data[mid] == val:
            a.next(mid)
            return count
        else:
            cur_cur = mid
            mid = (cur_cur + prev_cur) / 2
    
    # if not find, set cursor in list to cur_cur, since when it comes here, data[cur_cur] is the number 
    # closest to val and larger than val.
    a.next(cur_cur)
    return count  


###################################################################################################################
## Question No. 2:

def Logarithmic_merge(index, cut_off, buffer_size): # do not change the heading of the function
    pass # **replace** this line with your code





###################################################################################################################
## Question No. 3:

def decode_gamma(inputs):# do not change the heading of the function
    pass # **replace** this line with your code

def decode_delta(inputs):# do not change the heading of the function
    pass # **replace** this line with your code

def decode_rice(inputs, b):# do not change the heading of the function
    pass # **replace** this line with your code