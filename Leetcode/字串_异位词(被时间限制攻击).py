class Solution(object):
    def findAnagrams(self, s, p):
        start_index = 0
        end_index = 0
        answer = []
        width = len(p)
        while start_index <= (len(s)-width):
            temp_p = p
            while end_index < len(s) and s[end_index] in temp_p:
                temp_p = temp_p.replace(s[end_index],'',1)
                end_index += 1
            if end_index - start_index == width:
                answer += [start_index]
            start_index += 1
            end_index = start_index
        return answer
                
        
class1 = Solution()
condition_1 = "cbaebabacd",
print(len(condition_1))
condition_2 = "abc"
answer = class1.findAnagrams(condition_1,condition_2)
print(answer)