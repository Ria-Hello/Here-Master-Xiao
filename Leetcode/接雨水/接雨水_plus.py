class Solution(object):
    def trap(self, height):
        l_max = 0
        r_max = 0
        left_index = 0
        right_index = len(height)-1
        water = 0
        if len(height) == 1:
            return 0

        while left_index<right_index:
            l_wall = height[left_index]
            r_wall = height[right_index]

            if l_wall < r_wall:
                if l_wall>l_max:
                    l_max = l_wall
                else:
                    water += l_max - l_wall
                left_index += 1
            elif r_wall <= l_wall:
                if r_wall > r_max:
                    r_max = r_wall
                else:
                    water += r_max - r_wall
                right_index -= 1
        return water
            


class1 = Solution()
condition = [4,3,3,9,3,0,9,2,8,3]
answer = class1.trap(condition)
print(answer)
