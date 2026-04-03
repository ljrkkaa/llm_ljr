def partition(nums,left,right):
    pivot = nums[right]
    i = left
    for j in range(left,right):
        if nums[j] < pivot:
            nums[i],nums[j] = nums[j],nums[i]
            i+=1
    nums[right],nums[i] = nums[i],nums[right]

    return i


def quick_sort(nums, left,right):
    if left >= right:
        return

    i = partition(nums,left,right)

    quick_sort(nums,left,i-1)
    quick_sort(nums,i+1,right)


data = [10, 7, 8, 9, 1, 5]
quick_sort(data, 0, len(data) - 1)
print("排序结果:", data)