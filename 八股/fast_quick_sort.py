import random

class Solution:
    def sortArray(self, nums):
        self.quick_sort(nums, 0, len(nums) - 1)
        return nums

    def quick_sort(self, nums, left, right):
        while left < right:

            # 小数组优化：长度小于16用插入排序
            if right - left < 16:
                self.insertion_sort(nums, left, right)
                return

            pivot = nums[random.randint(left, right)]

            lt = left
            gt = right
            i = left

            # 三路划分
            while i <= gt:
                if nums[i] < pivot:
                    nums[i], nums[lt] = nums[lt], nums[i]
                    lt += 1
                    i += 1
                elif nums[i] > pivot:
                    nums[i], nums[gt] = nums[gt], nums[i]
                    gt -= 1
                else:
                    i += 1

            # 递归较小区间，较大区间用循环继续处理 尾递归消除（tail recursion elimination） 整体是o(log n)的递归深度
            if lt - left < right - gt:
                self.quick_sort(nums, left, lt - 1) # 递归处理较小区间
                left = gt + 1   # 循环处理较大区间
            else:
                self.quick_sort(nums, gt + 1, right)
                right = lt - 1

    def insertion_sort(self, nums, left, right):
        for i in range(left + 1, right + 1):
            key = nums[i]
            j = i - 1
            while j >= left and nums[j] > key:
                nums[j + 1] = nums[j]
                j -= 1
            nums[j + 1] = key