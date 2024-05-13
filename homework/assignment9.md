# Assignment #9: 图论：遍历，及 树算

Updated 1739 GMT+8 Apr 14, 2024

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

### 04081: 树的转换

http://cs101.openjudge.cn/dsapre/04081/



思路：



代码

```python
def tree_heights(s):
    old_height = 0
    max_old = 0
    new_height = 0
    max_new = 0
    stack = []
    for c in s:
        if c == 'd':
            old_height += 1
            max_old = max(max_old, old_height)

            new_height += 1
            stack.append(new_height)
            max_new = max(max_new, new_height)
        else:
            old_height -= 1

            new_height = stack[-1]
            stack.pop()
    return f"{max_old} => {max_new}"

s = input().strip()
print(tree_heights(s))

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-21 at 9.30.21 AM.png>)



### 08581: 扩展二叉树

http://cs101.openjudge.cn/dsapre/08581/



思路：



代码

```python
def build_tree(preorder):
    if not preorder or preorder[0] == '.':
        return None, preorder[1:]
    root = preorder[0]
    left, preorder = build_tree(preorder[1:])
    right, preorder = build_tree(preorder)
    return (root, left, right), preorder

def inorder(tree):
    if tree is None:
        return ''
    root, left, right = tree
    return inorder(left) + root + inorder(right)

def postorder(tree):
    if tree is None:
        return ''
    root, left, right = tree
    return postorder(left) + postorder(right) + root

preorder = input().strip()
tree, _ = build_tree(preorder)

print(inorder(tree))
print(postorder(tree))

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-21 at 9.38.28 AM.png>)



### 22067: 快速堆猪

http://cs101.openjudge.cn/practice/22067/



思路：



代码

```python
a = []
m = []

while True:
    try:
        s = input().split()
    
        if s[0] == "pop":
            if a:
                a.pop()
                if m:
                    m.pop()
        elif s[0] == "min":
            if m:
                print(m[-1])
        else:
            h = int(s[1])
            a.append(h)
            if not m:
                m.append(h)
            else:
                k = m[-1]
                m.append(min(k, h))
    except EOFError:
        break

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-21 at 9.25.59 AM.png>)



### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123



思路：



代码

```python
def is_valid_move(board_size, visited, row, col):
    return 0 <= row < board_size[0] and 0 <= col < board_size[1] and not visited[row][col]

def knight_tour(board_size, start_row, start_col):
    moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
             (1, 2), (1, -2), (-1, 2), (-1, -2)]

    visited = [[False] * board_size[1] for _ in range(board_size[0])]
    visited[start_row][start_col] = True
    count = [0]

    def dfs(row, col, visited, count):
        if all(all(row) for row in visited):
            count[0] += 1
            return

        for dr, dc in moves:
            next_row, next_col = row + dr, col + dc
            if is_valid_move(board_size, visited, next_row, next_col):
                visited[next_row][next_col] = True
                dfs(next_row, next_col, visited, count)
                visited[next_row][next_col] = False

    dfs(start_row, start_col, visited, count)
    return count[0]


T = int(input())

for _ in range(T):
    n, m, x, y = map(int, input().split())
    print(knight_tour((n, m), x, y))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-23 at 3.18.59 PM.png>)



### 28046: 词梯

bfs, http://cs101.openjudge.cn/practice/28046/



思路：



代码

```python
from collections import defaultdict

dic=defaultdict(list)
n,lis=int(input()),[]

for i in range(n):
    lis.append(input())

for word in lis:
    for i in range(len(word)):
        bucket=word[:i]+'_'+word[i+1:]
        dic[bucket].append(word)

def bfs(start,end,dic):
    queue=[(start,[start])]
    visited=[start]

    while queue:
        currentword,currentpath=queue.pop(0)

        if currentword==end:
            return ' '.join(currentpath)

        for i in range(len(currentword)):
            bucket=currentword[:i]+'_'+currentword[i+1:]

            for nbr in dic[bucket]:
                if nbr not in visited:
                    visited.append(nbr)
                    newpath=currentpath+[nbr]
                    queue.append((nbr,newpath))

    return 'NO'

start,end=map(str,input().split())    
print(bfs(start,end,dic)) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-21 at 9.41.19 AM.png>)



### 28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/



思路：



代码

```python
def is_valid_move(board_size, visited, row, col):
    return 0 <= row < board_size and 0 <= col < board_size and not visited[row][col]

def knight_tour(board_size, start_row, start_col):
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
             (1, -2), (1, 2), (2, -1), (2, 1)]

    visited = [[False] * board_size for _ in range(board_size)]
    visited[start_row][start_col] = True

    def get_neighbors(row, col):
        neighbors = []
        for dr, dc in moves:
            next_row, next_col = row + dr, col + dc
            if is_valid_move(board_size, visited, next_row, next_col):
                count = sum(1 for dr, dc in moves if is_valid_move(board_size, visited, next_row + dr, next_col + dc))
                neighbors.append((count, next_row, next_col))
        return neighbors

    def dfs(row, col, count):
        if count == board_size ** 2 - 1:
            return True

        neighbors = get_neighbors(row, col)
        neighbors.sort()

        for _, next_row, next_col in neighbors:
            visited[next_row][next_col] = True
            if dfs(next_row, next_col, count + 1):
                return True
            visited[next_row][next_col] = False

        return False

    return dfs(start_row, start_col, 0)

board_size = int(input())
start_row, start_col = map(int, input().split())
if knight_tour(board_size, start_row, start_col):
    print("success")
else:
    print("fail")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-23 at 3.16.11 PM-1.png>)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

题目较难，需要靠gpt协助完成。



