class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        """插入元素：末尾插入 + 上浮"""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        """删除堆顶：根节点与末尾交换 + 删除 + 下沉"""
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        
        root = self.heap[0]
        # 将最后一个元素放到堆顶
        self.heap[0] = self.heap.pop()
        # 从根节点开始下沉
        self._sift_down(0)
        return root

    def get_min(self):
        """获取堆顶最小值"""
        return self.heap[0] if self.heap else None

    def _sift_up(self, index):
        """上浮操作：将新元素移动到正确位置"""
        parent = (index - 1) // 2
        while index > 0 and self.heap[index] < self.heap[parent]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            index = parent
            parent = (index - 1) // 2

    def _sift_down(self, index):
        """下沉操作：将根元素移动到正确位置"""
        size = len(self.heap)
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index
            
            # 比较子节点
            if left < size and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < size and self.heap[right] < self.heap[smallest]:
                smallest = right
                
            if smallest != index:
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break

# 测试
mh = MinHeap()
elements = [5, 3, 8, 1, 2]
for e in elements:
    mh.push(e)
    print(f"Push {e}: {mh.heap}")

print("\nPop Min:")
while mh.heap:
    print(mh.pop(), end=" ")
