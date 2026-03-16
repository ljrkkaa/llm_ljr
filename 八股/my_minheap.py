class MinHeap:
    def __init__(self):
        self.heap = []

    def sift_up(self, idx):
        parent = (idx - 1) // 2
        while idx > 0 and self.heap[idx] < self.heap[parent]:
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            idx = parent
            parent = (idx - 1) // 2

    def sift_down(self, idx):
        n = len(self.heap)

        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx

            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left

            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest != idx:
                self.heap[idx], self.heap[smallest] = (
                    self.heap[smallest],
                    self.heap[idx],
                )
                idx = smallest
            else:
                break

    def build_heap(self, arr):
        self.heap = arr[:]
        start = (len(self.heap) - 2) // 2
        for i in range(start, -1, -1):
            self.sift_down(i)

    def push(self, num):
        self.heap.append(num)
        self.sift_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()

        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.sift_down(0)

        return min_val

    def get_min(self):
        return self.heap[0] if self.heap else None


mh = MinHeap()

arr = [5, 3, 8, 1, 2]
mh.build_heap(arr)

print(mh.heap)
