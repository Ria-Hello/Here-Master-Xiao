class Solution(object):
    def findAnagrams(self, s, p):
        start_index = 0
        answer = []
        end_index = 0
        p_changed = [0]*26
        temp_p = [0]*26
        for i in p:
            p_changed[ ord(i) - ord('a') ] += 1
        width = len(p)
        while start_index <= len(s) - width:
            if end_index < len(s):
                temp_p[(ord(s[end_index]) - ord('a'))] += 1
            if end_index - start_index > width-1:
                temp_p[ord(s[start_index]) - ord('a')] -= 1
                start_index += 1
            end_index += 1
            if temp_p == p_changed:
                answer += [start_index]
        return answer