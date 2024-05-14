# Assignment #D: May月考

Updated 1654 GMT+8 May 8, 2024

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

### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



思路：



代码

```python
L,M = map(int,input().split())
result = []
for _ in range(M):
    start,end = map(int,input().split())
    for i in range(start,end+1):
        result.append(i)

trees = set(result)        
print(L-len(trees)+1)

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-08 at 9.22.30 PM.png>)



### 20449: 是否被5整除

http://cs101.openjudge.cn/practice/20449/



思路：



代码

```python
A = str(input())
result = []
for i in range(1,len(A)+1):
    check = int(A[:i],2) #二进制转十进制
    if check % 5 == 0 or check == 5:
        result.append(1)
    else:
        result.append(0)
        
print(*result,sep='')

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-08 at 10.18.56 PM.png>)



### 01258: Agri-Net

http://cs101.openjudge.cn/practice/01258/



思路：



代码

```python
import heapq

def prim(graph):
    n = len(graph)
    visited = [False] * n
    min_heap = [(0, 0)]  #(distance, node)
    min_fiber = 0

    while min_heap:
        dist, node = heapq.heappop(min_heap)
        if visited[node]:
            continue
        visited[node] = True
        min_fiber += dist
        for neighbor, weight in enumerate(graph[node]):
            if not visited[neighbor] and weight != 0:
                heapq.heappush(min_heap, (weight, neighbor))

    return min_fiber

def main():
    while True:
        try:
            n = int(input())
            graph = []
            for _ in range(n):
                graph.append(list(map(int, input().split())))
            min_fiber = prim(graph)
            print(min_fiber)
        except EOFError:
            break

if __name__ == "__main__":
    main()

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-11 at 11.36.25 PM.png>)



### 27635: 判断无向图是否连通有无回路(同23163)

http://cs101.openjudge.cn/practice/27635/



思路：



代码

```python
def connected(graph,n):
    visited = [False] * n
    stack = [0]
    visited[0] = True
    
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                stack.append(neighbor)
                visited[neighbor] = True
    return all(visited)
                
def cycle(graph,n):
    def dfs(node,visited,parent):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor,visited,node):
                    return True
            elif parent != neighbor:
                return True
        return False
    
    visited = [False] * n
    for node in range(n):
        if not visited[node]:
            if dfs(node,visited,-1):
                return True
    return False
    

n,m = map(int,input().split())
graph = [[]for _ in range(n)]
for _ in range(m):
    u,v = map(int,input().split())
    graph[u].append(v)
    graph[v].append(u)
    
if connected(graph,n):
    print('connected:yes')
else:
    print('connected:no')
    
if cycle(graph,n):
    print('loop:yes')
else:
    print('loop:no') 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-09 at 3.29.26 PM.png>)





### 27947: 动态中位数

http://cs101.openjudge.cn/practice/27947/



思路：



代码

```python
import heapq

def solve(nums):
    max_heap = []
    min_heap = []
    
    median = []
    
    for i,num in enumerate(nums):
        if not max_heap or num <= -max_heap[0]:  #heap的性质是默认最小堆，所以在最大堆中要取负数
            heapq.heappush(max_heap,-num)
        else:
            heapq.heappush(min_heap,num)
            
        if len(max_heap) - len(min_heap) > 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap,-heapq.heappop(min_heap))
        
        if i % 2 == 0:
            median.append(-max_heap[0])
            
    return median
        
T = int(input())
for _ in range(T):
    nums = list(map(int, input().split()))
    median = solve(nums)
    print(len(median))
    print(*median)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-12 at 11.34.09 AM.png>)



### 28190: 奶牛排队

http://cs101.openjudge.cn/practice/28190/



思路：



代码

```python
N = int(input())
heights = [int(input()) for _ in range(N)]

left_bound = [-1] * N
right_bound = [N] * N

stack = []

for i in range(N):
    while stack and heights[stack[-1]] < heights[i]:
        stack.pop()

    if stack:
        left_bound[i] = stack[-1]

    stack.append(i)

stack = []

for i in range(N-1, -1, -1):
    while stack and heights[stack[-1]] > heights[i]:
        stack.pop()

    if stack:
        right_bound[i] = stack[-1]

    stack.append(i)

ans = 0

for i in range(N): 
    for j in range(left_bound[i] + 1, i):
        if right_bound[j] > i:
            ans = max(ans, i - j + 1)
            break
print(ans)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-12 at 12.54.45 PM.png>)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

通过这次作业有更加了解堆的做法，整体感觉比前几次的作业容易一些。



