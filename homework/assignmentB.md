# Assignment #B: 图论和树算

Updated 1709 GMT+8 Apr 28, 2024

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

### 28170: 算鹰

dfs, http://cs101.openjudge.cn/practice/28170/



思路：



代码

```python
def dfs(board, visited, row, col):
    if row < 0 or row >= 10 or col < 0 or col >= 10 or visited[row][col] or board[row][col] == '-':
        return
    visited[row][col] = True
    dfs(board, visited, row + 1, col)
    dfs(board, visited, row - 1, col)
    dfs(board, visited, row, col + 1)
    dfs(board, visited, row, col - 1)

def count_eagles(board):
    visited = [[False] * 10 for _ in range(10)]
    eagles = 0
    for i in range(10):
        for j in range(10):
            if not visited[i][j] and board[i][j] == '.':
                dfs(board, visited, i, j)
                eagles += 1
    return eagles

board = []
for _ in range(10):
    row = input().strip()
    board.append(row)
print(count_eagles(board))

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-05 at 1.46.36 PM.png>)



### 02754: 八皇后

dfs, http://cs101.openjudge.cn/practice/02754/



思路：



代码

```python
def check(board, row, col):
    for i in range(row):
        if board[i] == col or abs(row - i) == abs(col - board[i]):
            return False
    return True

def solve(board, row, count, target):
    if row == 8:
        count[0] += 1
        if count[0] == target:
            print("".join(str(col + 1) for col in board))
            return True
    else:
        for col in range(8):
            if check(board, row, col):
                board[row] = col
                if solve(board, row + 1, count, target):
                    return True
    return False

def find_queen_sequence(b):
    board = [0] * 8
    count = [0]
    solve(board, 0, count, b)

n = int(input())
for _ in range(n):
    b = int(input())
    find_queen_sequence(b)

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-05 at 2.00.31 PM.png>)



### 03151: Pots

bfs, http://cs101.openjudge.cn/practice/03151/



思路：



代码

```python
def bfs(A, B, C):
    start = (0, 0)
    visited = set()
    visited.add(start)
    queue = [(start, [])]

    while queue:
        (a, b), actions = queue.pop(0)

        if a == C or b == C:
            return actions

        next_states = [(A, b), (a, B), (0, b), (a, 0), (min(a + b, A),\
                max(0, a + b - A)), (max(0, a + b - B), min(a + b, B))]

        for i in next_states:
            if i not in visited:
                visited.add(i)
                new_actions = actions + [get_action(a, b, i)]
                queue.append((i, new_actions))

    return ["impossible"]


def get_action(a, b, next_state):
    if next_state == (A, b):
        return "FILL(1)"
    elif next_state == (a, B):
        return "FILL(2)"
    elif next_state == (0, b):
        return "DROP(1)"
    elif next_state == (a, 0):
        return "DROP(2)"
    elif next_state == (min(a + b, A), max(0, a + b - A)):
        return "POUR(2,1)"
    else:
        return "POUR(1,2)"


A, B, C = map(int, input().split())
solution = bfs(A, B, C)

if solution == ["impossible"]:
    print(solution[0])
else:
    print(len(solution))
    for i in solution:
        print(i)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-05 at 2.03.04 PM.png>)



### 05907: 二叉树的操作

http://cs101.openjudge.cn/practice/05907/



思路：



代码

```python
class BinaryTree:
    def __init__(self,root):
        self.root = root
        self.left = None
        self.right = None
        self.father = None

for _ in range(int(input())):
    n,m = map(int,input().split())
    tree_list = list(BinaryTree(i) for i in range(n))

    for __ in range(n):
        root,left,right = map(int,input().split())
        if left != -1:
            tree_list[root].left = tree_list[left]
            tree_list[left].father = tree_list[root]
        if right != -1:
            tree_list[root].right = tree_list[right]
            tree_list[right].father = tree_list[root]

    for __ in range(m):
        type,*tu = map(int,input().split())

        if type == 1: 
            x,y = tu
            tree1,tree2 = tree_list[x],tree_list[y]
            father1 = tree1.father
            father2 = tree2.father
            if father2 is father1:
                father2.left,father2.right = father2.right,father2.left

            else:
                if father1.left == tree1:
                    father1.left = tree2
                else: father1.right = tree2

                if father2.left == tree2:
                    father2.left = tree1
                else: father2.right = tree1
                tree1.father,tree2.father = father2,father1

        elif type == 2:
            node = tree_list[tu[0]]
            while node.left:
                node = node.left
            print(node.root)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-05 at 2.11.18 PM.png>)





### 18250: 冰阔落 I

Disjoint set, http://cs101.openjudge.cn/practice/18250/



思路：



代码

```python

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x

while True:
    try:
        n, m = map(int, input().split())
        parent = list(range(n + 1))

        for _ in range(m):
            a, b = map(int, input().split())
            if find(a) == find(b):
                print('Yes')
            else:
                print('No')
                union(a, b)

        unique_parents = set(find(x) for x in range(1, n + 1)) 
        ans = sorted(unique_parents) 
        print(len(ans))
        print(*ans)

    except EOFError:
        break

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-05 at 2.16.55 PM.png>)



### 05443: 兔子与樱花

http://cs101.openjudge.cn/practice/05443/



思路：



代码

```python
import heapq

def dijkstra(adjacency, start):
    distances = {vertex: float('infinity') for vertex in adjacency}
    previous = {vertex: None for vertex in adjacency}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in adjacency[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous

def shortest_path_to(adjacency, start, end):
    distances, previous = dijkstra(adjacency, start)
    path = []
    current = end
    while previous[current] is not None:
        path.insert(0, current)
        current = previous[current]
    path.insert(0, start)
    return path, distances[end]

P = int(input())
places = {input().strip() for _ in range(P)}

Q = int(input())
graph = {place: {} for place in places}
for _ in range(Q):
    src, dest, dist = input().split()
    dist = int(dist)
    graph[src][dest] = dist
    graph[dest][src] = dist  

R = int(input())
requests = [input().split() for _ in range(R)]

for start, end in requests:
    if start == end:
        print(start)
        continue

    path, total_dist = shortest_path_to(graph, start, end)
    output = ""
    for i in range(len(path) - 1):
        output += f"{path[i]}->({graph[path[i]][path[i+1]]})->"
    output += f"{end}"
    print(output)


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-05-05 at 2.20.17 PM.png>)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

除了前两题，后面的题还是比较有难度的。花了较长时间才完成。



