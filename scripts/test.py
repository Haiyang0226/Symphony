'''
题目要求：
给定一个整数数组 nums，找到其中最长的严格递增子序列，并返回该子序列的和。如果有多个最长递增子序列，返回其中和最大的那个。

示例1：

输入：nums = [1, 3, 2, 4]

输出：8

解释：最长的严格递增子序列是 [1, 3, 4] 或 [1, 2, 4]，它们的长度都是 3，但 [1, 3, 4] 的和为 8，是最大的。

示例2：

输入：nums = [5, 4, 3, 2, 1]

输出：5

解释：最长的严格递增子序列是 [5]，长度为 1，和为 5。

'''
import numpy

#XXXXXXX j   XXXXX i

nums=[5, 4, 3, 2, 1]
#nums=[1,3,2,4]
n=len(nums)
dp=[0]*n
length=[1]*n

for i in range(n):
    dp[i]=nums[i]

for i in range (1,n):
    for j in range(i):
        if nums[j]<nums[i]:
            print(i,j,nums[j],nums[i])
            if length[j]+1>length[i] or (length[j]+1==length[i] and dp[j]+nums[i]>dp[i]):
                #print(i,j)
                length[i]=length[j]+1
                dp[i]=dp[j]+nums[i]
max_length=max(length)
max_sum=0

for i in range(n):
    if length[i]==max_length:
        max_sum=max(max_sum,dp[i])
print(max_sum)

