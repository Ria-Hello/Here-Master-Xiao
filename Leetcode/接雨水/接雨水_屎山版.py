class Solution(object):
    def trap(self, height):
        answer1 = 0
        answer2 = 0
        taken = 0
        start_pos = 0
        final_pos = 0
        max_height = 0
        max_index = 0
        answer_change = 0
        flag = 0
        max_time = 0
        loss = 0
        length = len(height)-1
        for i in height:
            if i>max_height:
                max_height = i
        for i in height[1:length]:
            if i == max_height:
                max_time += 1
        while start_pos < length and (height[start_pos]==0 or height[start_pos + 1] > height[start_pos]):
            start_pos += 1
            final_pos += 1
        if height[start_pos] == max_height:
            flag = 1
        while final_pos < length:
            taken = 0
            final_pos += 1
            S_flag = 0
            if height[final_pos]>=height[start_pos]:
                width = final_pos - start_pos - 1
                tall = min(height[final_pos],height[start_pos])
                taken += sum(height[start_pos + 1:final_pos])
                temp_answer = width*tall - taken
                if temp_answer > 0:
                    S_flag = 1
                    answer1 += temp_answer
                    if max_time >1 and (height[start_pos]==max_height) and (height[final_pos] == max_height):
                        loss += temp_answer
                if S_flag == 1:
                    start_pos = final_pos
                if final_pos == (start_pos + 1):
                    start_pos = final_pos
                
        final_pos = length
        start_pos = length
        while start_pos > 0 and (height[start_pos] == 0 or height[start_pos-1] > height[start_pos]):
            start_pos -= 1
            final_pos -= 1
        if flag == 1 and height[start_pos] == max_height:
            flag = 2
        elif height[start_pos] == max_height:
            flag = 3
        while final_pos >0:
            taken = 0
            final_pos -= 1
            S_flag = 0
            if height[final_pos]>=height[start_pos]:
                width = -final_pos + start_pos - 1
                tall = min(height[final_pos],height[start_pos])
                taken += sum(height[final_pos + 1:start_pos])
                temp_answer = width*tall - taken
                if temp_answer > 0:
                    S_flag = 1
                    answer2 += temp_answer
                    if max_time >1 and (height[start_pos]==max_height) and (height[final_pos] == max_height):
                        loss += temp_answer
                if S_flag == 1:
                    start_pos = final_pos
                if final_pos == start_pos-1:
                    start_pos = final_pos  
        if max_time == 1 and flag == 0:
            return (answer1+answer2)
        elif max_time>1:
            return ((answer1+answer2)-int(loss/2))
        else:
            return max(answer1,answer2)
class1 = Solution()
condition = [4,3,3,9,3,0,9,2,8,3]
answer = class1.trap(condition)
print(answer)
