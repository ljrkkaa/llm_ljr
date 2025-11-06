from typing import List

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g = [[] for _ in range(numCourses)]
        for cur, pre in prerequisites:
            g[pre].append(cur)

        colors = [0] * numCourses

        def dfs(x):
            colors[x] = 1
            for y in g[x]:
                if colors[y]==1 or (colors[y] == 0 and dfs(y)):
                    return True
            colors[x] = 2
            return False

        for i, c in enumerate(colors):
            if c == 0 and dfs(i):
                return False
        return True



if __name__ == "__main__":
    # 示例 1：有环
    print("\n测试用例 1（有环）:")
    s = Solution()
    print(s.canFinish(4, [[1, 0], [2, 1], [0, 2], [3, 2]]))

    # 示例 2：无环
    print("\n测试用例 2（无环）:")
    s = Solution()
    print(s.canFinish(4, [[1, 0], [2, 1], [3, 2]]))
