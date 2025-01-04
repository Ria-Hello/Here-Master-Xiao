class Solution(object):
    def subarraySum(self, nums, k):
        sum = 0
        answer = 0
        dict = {}
        dict[0] = 1
        for num in nums:
            sum += num
            if (sum - k) in dict:
                answer += dict[sum-k]
            dict[sum] = dict.get(sum,0) + 1
        return answer