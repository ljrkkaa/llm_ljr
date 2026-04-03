def bubble_sort(arr):
    n = len(arr)

    for i in range(n - 1):
        cnt = 0
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                cnt += 1
        if cnt == 0:
            break
    return arr


if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90]
    print("Sorted array is:", bubble_sort(arr))
