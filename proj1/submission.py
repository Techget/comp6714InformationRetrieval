## Submission.py for COMP6714-Project1
import math

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

##### decode gamma code #####

def binary_str_to_int(string):
    s = string[::-1]
    base = 0
    output = 0
    
    for char in s:
        output += int(char) * (2 ** base)
        base += 1
    
    return output
    
def decode_gamma(inputs):# do not change the function heading
    if inputs == '':
        # In gamma code, not even a digit stands for 0
        return [0]
    
    binary_bits = 0
    unary_flag = True
    # The first binary '1' is deliberately omitted, so the string init with '1' instead of ''
    binary_string = '1'
    output = []
    
    for bit in inputs:
        if unary_flag:
            if bit == '1':
                binary_bits += 1
            else:
                unary_flag = False
        else:
            if binary_bits > 0:
                # concatenate to binary_string
                binary_string += bit
                binary_bits -= 1
            else:
                output.append(binary_str_to_int(binary_string))
                # recover for next number decoding
                binary_string = '1'
                if bit == '0':
                    # means next number is 1
                    binary_bits = 0
                    unary_flag = False
                elif bit == '1':
                    binary_bits = 1
                    unary_flag = True
                else:
                    print("Strange, the bit is:", bit)
    
    # The last number
    output.append(binary_str_to_int(binary_string))
    
    return output

##### spliter, decode delta code #####

def read_one_delta_number(input_string):
    if input_string[0] == '0':
        return 1,1
    
    binary_bits = 0
    unary_flag = True
    output = None
    gamma_code = ''
    cur_pos = 0
    
    for bit in input_string:
        if unary_flag:
            if bit == '1':
                binary_bits += 1
                gamma_code += bit
            else:
                gamma_code += bit
                unary_flag = False
        else:
            if binary_bits > 0:
                gamma_code += bit
                binary_bits -= 1
            else:
                # make use of the `decode_gamma` we defined previously
                digits_need = decode_gamma(gamma_code)[0]
                # The highest radix 1 is removed, so we need to do minus 1 here
                digits_need -= 1
                
                binary_string = '1' + input_string[cur_pos:cur_pos + digits_need]
                # print("binary_string:", binary_string)
                output = binary_str_to_int(binary_string)
                
                cur_pos += digits_need
                break
        
        cur_pos += 1
    
    # return the how many digits has read, also the corresponding number
    return cur_pos, output

def decode_delta(inputs):# do not change the function heading
    cur_pos = 0
    output = []
    
    while cur_pos < len(inputs):  
        temp_cur_pos, temp_output = read_one_delta_number(inputs[cur_pos:])
        cur_pos += temp_cur_pos
        output.append(temp_output)

    return output

##### spliter, decode rice code #####

def read_one_rice_number(input_string, b):
    digits_after_unary = int(math.log(b, 2))
    
    #special case for 1
    if input_string[0:(1 + digits_after_unary)] == '0' * (1+digits_after_unary):
        return 1 + digits_after_unary, 1
   
    unary_flag = True
    output = None
    unary_number = 0
    cur_pos = 0
    
    for bit in input_string:
        if unary_flag:
            if bit == '1':
                unary_number += 1
            else:
                unary_flag = False
        else:
            binary_string = input_string[cur_pos : cur_pos + digits_after_unary]
            r = binary_str_to_int(binary_string)
            output = unary_number * b + r
            cur_pos += digits_after_unary
            break
        
        cur_pos += 1
    
    # return the how many digits has read, also the corresponding number
    return cur_pos, output

def decode_rice(inputs, b):# do not change the function heading
    cur_pos = 0
    output = []
    
    while cur_pos < len(inputs):  
        temp_cur_pos, temp_output = read_one_rice_number(inputs[cur_pos:], b)
        cur_pos += temp_cur_pos
        output.append(temp_output)

    return output