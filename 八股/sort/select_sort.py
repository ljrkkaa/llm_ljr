def select_sort(arr):
    n = len(arr)

    for i in range(n-1):
        min_idx = i
        for j in range(i+1,n):
            if arr[min_idx] > arr[j]:
                min_idx = j
        
        arr[min_idx], arr[i] = arr[i], arr[min_idx]
    
    return arr

arr = [3,1,4,6,6]
print(select_sort(arr))
