## Submission.py for COMP6714-Project1

###################################################################################################################
## Question No. 0:
def add(a, b): # do not change the heading of the function
    return a + b




###################################################################################################################
## Question No. 1:

def gallop_to(a, val):# do not change the heading of the function
    count = 0
    inc = 1
    cur_cur = a.cur
    
    ### first step: gallop ###
    gallop_val = a.peek(cur_cur)
    # the case where it starts with large value
    if gallop_val >= val:
        # print("invalid")
        return count
    
    while gallop_val < val:
        gallop_val = a.peek(cur_cur)
        # print("in first while loop, gallop_val:", gallop_val)
        count += 1
        # it comes to an end. but still smaller than val, return directly, the wrap function will deal with this
        if gallop_val == None:
            return count
        
        if gallop_val < val:
            a.next(val = inc)
            cur_cur = a.cur
            inc *= 2
        else:
            break
    
    ### if currently, a.data[cur] is equal to val, then we do not need to do BS
    if a.peek(cur_cur) == val:
        return count

    ### binary search ###
    cur_cur = a.cur
    prev_cur = cur_cur - inc//2
    # print("binary search start: cur_cur:",cur_cur, "prev_cur:", prev_cur)
    
    # do not need to make a copy, since no modification will be made
    data = a.data
    mid = (cur_cur + prev_cur) // 2
    
    while cur_cur - prev_cur > 1:
        if data[mid] < val:
            prev_cur = mid
            mid = (cur_cur + prev_cur) // 2
        elif data[mid] == val:
            # a.next(mid)
            a.cur = mid
            return count
        else:
            cur_cur = mid
            mid = (cur_cur + prev_cur) // 2
    
    # if not find, set cursor in list to cur_cur, since when it comes here, data[cur_cur] is the number 
    # closest to val and larger than val.
    # a.next(cur_cur)
    a.cur = cur_cur
    # print("execute to end, cur_cur: ", cur_cur)
    # print(a.cur)
    return count    



###################################################################################################################
## Question No. 2:

def merge_two_list(l1, l2):
    data = l1[:]
    data.extend(l2)
    data.sort()
    return data

def Logarithmic_merge(index, cut_off, buffer_size): # do not change the heading of the function
    buffer = []
    output = []
    # highest_gen = 0
    
    for i in range(cut_off):
        cur_index = index[i]
        
        # put into buffer
        if len(buffer) < buffer_size:
            buffer.append(cur_index)
            continue
        
        # need to merge
        # this buffer will be used as intermediate data structure in the following loop as well
        buffer.sort()
        merge_flag = False
        temp_gen = 0
        while merge_flag == False:
            if temp_gen >= len(output):
                # means we need a new generation
                output.append(buffer)
                merge_flag = True
            elif output[temp_gen] == []:
                # this generation is empty put buffer here
                output[temp_gen] = buffer
                merge_flag = True
            else:
                # actually need to do merge
                buffer = merge_two_list(buffer, output[temp_gen])
                output[temp_gen] = []
                temp_gen += 1          
    
        # clean up for next index
        buffer = [cur_index]
        
    # last buffer still need to put into consideration
    if buffer != []:
        output.insert(0, buffer)
    else:
        output.insert(0, [])

    return output



###################################################################################################################
## Question No. 3:

def decode_gamma(inputs):# do not change the heading of the function
    pass # **replace** this line with your code

def decode_delta(inputs):# do not change the heading of the function
    pass # **replace** this line with your code

def decode_rice(inputs, b):# do not change the heading of the function
    pass # **replace** this line with your code