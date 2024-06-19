# Assignment #F: All-Killed 满分

Updated 1844 GMT+8 May 20, 2024

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

### 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/



思路：bfs



代码

```python
from collections import deque

def right_view(n, tree):
    queue = deque([(1, tree[1])])  
    right_view = []

    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node, children = queue.popleft()
            if children[0] != -1:
                queue.append((children[0], tree[children[0]]))
            if children[1] != -1:
                queue.append((children[1], tree[children[1]]))
        right_view.append(node)

    return right_view

n = int(input())
tree = {1: [-1, -1] for _ in range(n+1)}  
for i in range(1, n+1):
    left, right = map(int, input().split())
    tree[i] = [left, right]

result = right_view(n, tree)
print(' '.join(map(str, result)))

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-21 at 5.51.27 PM.png>)



### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/



思路：



代码

```python
n = int(input())
a = list(map(int,input().split()))

stack = []
for i in range(n):
    while stack and a[stack[-1]] < a[i]:
        a[stack.pop()] = i + 1

    stack.append(i)

while stack:
    a[stack[-1]] = 0
    stack.pop()
    
print(*a)

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-22 at 2.42.38 PM.png>)



### 09202: 舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/



思路：



代码

```python
from collections import deque

def detect_cycle_in_directed_graph():
    T = int(input())
    for _ in range(T):
        N, M = map(int, input().split())
        graph = [[] for _ in range(N + 1)]
        in_degree = [0] * (N + 1)
        
        for _ in range(M):
            u, v = map(int, input().split())
            graph[u].append(v)
            in_degree[v] += 1
        
        def has_cycle(graph, in_degree, n):
            queue = deque()
            for i in range(1, n + 1):
                if in_degree[i] == 0:
                    queue.append(i)
            
            visited_count = 0
            while queue:
                node = queue.popleft()
                visited_count += 1
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            return visited_count != n
        
        if has_cycle(graph, in_degree, N):
            print('Yes')
        else:
            print('No')

detect_cycle_in_directed_graph()


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-23 at 6.30.53 PM-1.png>)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



思路：看了题解和gpt才明白...



代码

```python
n,m = map(int,input().split())
L = list(int(input()) for x in range(n))

def check(x):
    num, cut = 1, 0
    for i in range(n):
        if cut + L[i] > x:
            num += 1
            cut = L[i]  
        else:
            cut += L[i]
    
    if num > m:
        return False
    else:
        return True

maxmax = sum(L)
minmax = max(L)
while minmax < maxmax:
    middle = (maxmax + minmax) // 2
    if check(middle):   #表明这种插法可行，那么看看更小的插法可不可以
        maxmax = middle
    else:
        minmax = middle + 1#这种插法不可行，改变minmax看看下一种插法可不可以

print(maxmax)   

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-23 at 8.28.53 PM.png>)



### 07735: 道路

http://cs101.openjudge.cn/practice/07735/



思路：



代码

```python
import heapq

def shortest_path_with_fee(K, N, R, roads):
    # 使用一个三维数组来记录到达某个城市花费某个金币时的最短路径
    dp = [[float('inf')] * (K + 1) for _ in range(N + 1)]
    dp[1][0] = 0  # 起点城市1，花费0金币，路径长度为0

    # 优先队列（路径长度，当前城市，花费）
    heap = [(0, 1, 0)]  # (current length, current city, current cost)

    while heap:
        current_length, current_city, current_cost = heapq.heappop(heap)
        
        if current_city == N:
            return current_length
        
        if current_length > dp[current_city][current_cost]:
            continue
        
        for s, d, l, t in roads:
            if s == current_city and current_cost + t <= K:
                new_cost = current_cost + t
                new_length = current_length + l
                if new_length < dp[d][new_cost]:
                    dp[d][new_cost] = new_length
                    heapq.heappush(heap, (new_length, d, new_cost))

    return -1

K = int(input().strip())
N = int(input().strip())
R = int(input().strip())

roads = []
for _ in range(R):
    S, D, L, T = map(int, input().split())
    roads.append((S, D, L, T))

result = shortest_path_with_fee(K, N, R, roads)
print(result)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-23 at 8.44.35 PM.png>)



### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



思路：



代码

```python
def find(x):    # 并查集查询
    if p[x] == x:
        return x
    else:
        p[x] = find(p[x])  # 父节点设为根节点。目的是路径压缩。
        return p[x]

n, k = map(int, input().split())

p = [0] * (3 * n + 1)
for i in range(3 * n + 1):  # 并查集初始化
    p[i] = i

ans = 0
for _ in range(k):
    a, x, y = map(int, input().split())
    if x > n or y > n:
        ans += 1
        continue
    
    if a == 1:
        if find(x + n) == find(y) or find(y + n) == find(x):
            ans += 1
            continue
        
        # 合并
        p[find(x)] = find(y)
        p[find(x + n)] = find(y + n)
        p[find(x + 2 * n)] = find(y + 2 * n)
    else:
        if find(x) == find(y) or find(y + n) == find(x):
            ans += 1
            continue
        p[find(x + n)] = find(y)
        p[find(y + 2 * n)] = find(x)
        p[find(x + 2 * n)] = find(y + n)

print(ans)
 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-23 at 9.10.02 PM.png>)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

对我还是比较有难度，希望机考的难度可以比这次作业稍低一点。



