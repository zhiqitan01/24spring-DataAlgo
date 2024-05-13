# Assignment #8: 图论：概念、遍历，及 树算

Updated 1150 GMT+8 Apr 8, 2024

2024 spring, Complied by 陈紫琪，信息管理系



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS Ventura 13.4.1 (c)

Python编程环境：Spyder IDE 5.2.2



## 1. 题目

### 19943: 图的拉普拉斯矩阵

matrices, http://cs101.openjudge.cn/practice/19943/



思路：



代码

```python
n,m = map(int,input().split())
matrix_d = [[0]*n for i in range(n)]
matrix_a = [[0]*n for i in range(n)]

for _ in range(m):
    a,b = map(int,input().split())
    matrix_d[a][a] += 1
    matrix_d[b][b] += 1
    matrix_a[a][b] += 1
    matrix_a[b][a] += 1
    
result = [[matrix_d[i][j] - matrix_a[i][j] for j in range(n)] for i in range(n)]

for i in range(n):
    print(*result[i])

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-09 at 3.43.39 PM.png>)



### 18160: 最大连通域面积

matrix/dfs similar, http://cs101.openjudge.cn/practice/18160



思路：



代码

```python
def max_connected_area(grid, n, m, i, j):
    if i < 0 or i >= n or j < 0 or j >= m or grid[i][j] != 'W':
        return 0
    
    grid[i][j] = '.'  
    
    result = 1  
    for x in range(-1, 2):
        for y in range(-1, 2):
            result += max_connected_area(grid, n, m, i + x, j + y)

    return result

def find_largest_connected_area(T, data):
    for _ in range(T):
        n, m = data[_][0]
        grid = data[_][1]

        max_area = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 'W':
                    area = max_connected_area(grid, n, m, i, j)
                    max_area = max(max_area, area)

        print(max_area)

if __name__ == "__main__":
    T = int(input())
    data = []

    for _ in range(T):
        n, m = map(int, input().split())
        grid = [list(input()) for _ in range(n)]
        data.append(((n, m), grid))

    find_largest_connected_area(T, data)

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-10 at 1.50.15 PM.png>)



### sy383: 最大权值连通块

https://sunnywhy.com/sfbj/10/3/383



思路：



代码

```python
def max_weight(n, m, weights, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v) #把每个顶点连接到的顶点的值储存起来
        graph[v].append(u) #因为无向图的边是双向的

    visited = [False] * n
    max_weight = 0

    def dfs(node):
        visited[node] = True #node表示当前遍历到的顶点
        total_weight = weights[node] #初始化total_weight为当前顶点的权值
        for neighbor in graph[node]:
            if not visited[neighbor]:
                total_weight += dfs(neighbor) 
        return total_weight

    for i in range(n):
        if not visited[i]:
            max_weight = max(max_weight, dfs(i))

    return max_weight

n, m = map(int, input().split())
weights = list(map(int, input().split()))
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

print(max_weight(n, m, weights, edges))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-16 at 2.26.05 PM.png>)



### 03441: 4 Values whose Sum is 0

data structure/binary search, http://cs101.openjudge.cn/practice/03441



思路：



代码

```python
def count_quadruplets_with_zero_sum(A, B, C, D):
    sums_AB = {}
    count = 0

    for a in A:
        for b in B:
            sums_AB[a + b] = sums_AB.get(a + b, 0) + 1

    for c in C:
        for d in D:
            complement = -(c + d)
            if complement in sums_AB:
                count += sums_AB[complement]

    return count

n = int(input())
A = []
B = []
C = []
D = []

for _ in range(n):
    nums = list(map(int, input().split()))
    A.append(nums[0])
    B.append(nums[1])
    C.append(nums[2])
    D.append(nums[3])

result = count_quadruplets_with_zero_sum(A, B, C, D)
print(result)


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-10 at 2.35.43 PM.png>)



### 04089: 电话号码

trie, http://cs101.openjudge.cn/practice/04089/



思路：



代码

```python
class TrieNode:
    def __init__(self):
        self.child={}


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, nums):
        curnode = self.root
        for x in nums:
            if x not in curnode.child:
                curnode.child[x] = TrieNode()
            curnode=curnode.child[x]

    def search(self, num):
        curnode = self.root
        for x in num:
            if x not in curnode.child:
                return 0
            curnode = curnode.child[x]
        return 1


t = int(input())
p = []
for _ in range(t):
    n = int(input())
    nums = []
    for _ in range(n):
        nums.append(str(input()))
    nums.sort(reverse=True)
    s = 0
    trie = Trie()
    for num in nums:
        s += trie.search(num)
        trie.insert(num)
    if s > 0:
        print('NO')
    else:
        print('YES') 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-16 at 2.34.51 PM.png>)



### 04082: 树的镜面映射

http://cs101.openjudge.cn/practice/04082/



思路：



代码

```python
from collections import deque

class TreeNode:
    def __init__(self, x):
        self.x = x
        self.children = []

def create_node():
    return TreeNode('')

def build_tree(tempList, index):
    node = create_node()
    node.x = tempList[index][0]
    if tempList[index][1] == '0':
        index += 1
        child, index = build_tree(tempList, index)
        node.children.append(child)
        index += 1
        child, index = build_tree(tempList, index)
        node.children.append(child)
    return node, index

def print_tree(p):
    Q = deque()
    s = deque()

    # 遍历右子节点并将非虚节点加入栈s
    while p is not None:
        if p.x != '$':
            s.append(p)
        p = p.children[1] if len(p.children) > 1 else None

    # 将栈s中的节点逆序放入队列Q
    while s:
        Q.append(s.pop())

    # 宽度优先遍历队列Q并打印节点值
    while Q:
        p = Q.popleft()
        print(p.x, end=' ')

        # 如果节点有左子节点，将左子节点及其右子节点加入栈s
        if p.children:
            p = p.children[0]
            while p is not None:
                if p.x != '$':
                    s.append(p)
                p = p.children[1] if len(p.children) > 1 else None

            # 将栈s中的节点逆序放入队列Q
            while s:
                Q.append(s.pop())


n = int(input())
tempList = input().split()
root, _ = build_tree(tempList, 0)
print_tree(root)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-16 at 2.37.26 PM.png>)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==
题目较难，对无向图的概念不太熟悉，需要依靠gpt的协助完成作业。




