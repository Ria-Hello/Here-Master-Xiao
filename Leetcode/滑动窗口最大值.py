class Solution(object):
    def maxSlidingWindow(self, nums, k):
        answer = []
        index_list = []
        start_index = 0
        for i in range(len(nums) ):
            start_index = i - k + 1
            index_list.append(i)
            if index_list and start_index > 0:
                index_list.pop(0)
            if nums[i] < index_list[-1]:
                index_list.pop()
            if start_index >= 0:
                answer += [nums[index_list[0]]]
        return answer
            
                
        
class1 = Solution()
condition_1  = [1,1,1]
condition_2 = 2
answer = class1.maxSlidingWindow(condition_1,condition_2)
print(answer)



#####
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        answer = []
        index_list = []
        start_index = 0
        for i in range(len(nums) ):
            index_list.append(i)
            start_index = i - k + 1
            if index_list and start_index > 0:
                index_list.pop(0)
            if index_list and nums[i] < nums[index_list[0]]:
                print(index_list)
                print('\n')
                index_list.pop()
            if index_list and nums[i] > nums[index_list[0]]:
                index_list.pop(0)
            if start_index >= 0:
                answer += [nums[index_list[0]]]
        return answer
            
                
        
class1 = Solution()
condition_1  = [1,3,-1,-3,5,3,6,7]
condition_2 =3
answer = class1.maxSlidingWindow(condition_1,condition_2)
print(answer)