# Assignment #4: 排序、栈、队列和树

Updated 0005 GMT+8 March 11, 2024

2024 spring, Complied by 陈紫琪，信息管理系



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:

Learn about Time complexities, learn the basics of individual Data Structures, learn the basics of Algorithms, and practice Problems.

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS Ventura 13.4.1 (c)

Python编程环境：Spyder IDE 5.2.2



## 1. 题目

### 05902: 双端队列

http://cs101.openjudge.cn/practice/05902/



思路：



代码

```python
from collections import deque

t = int(input())
for _ in range(t):
    n = int(input())
    a = []
    for _ in range(n):
        x,y = input().split()
        a.append((int(x),int(y)))

    queue = deque()        
    for i in a:
        if i[0] == 1:
            queue.append(i[1])
        elif i[0] == 2:
            if i[1] == 0:
                queue.popleft()
            elif i[1] == 1:
                queue.pop()

    if len(queue) == 0:
        print("NULL")
    else:            
        print(" ".join(map(str,queue)))

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-12 at 5.31.23 PM.png>)



### 02694: 波兰表达式

http://cs101.openjudge.cn/practice/02694/



思路：



代码

```python
expression = input().split()
stack = []

while expression:
    a = expression.pop(-1)
    if a in ['+', '-', '*', '/']:
        b = stack.pop(-1)
        c = stack.pop(-1)
        if a == '+':
            stack.append(b + c)
        elif a == '-':
            stack.append(b - c)
        elif a == '*':
            stack.append(b * c)
        else:
            stack.append(b / c)
    else:
        stack.append(float(a))
        
print("{:.6f}".format(stack[0]))

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-12 at 10.22.12 PM.png>)



### 24591: 中序表达式转后序表达式

http://cs101.openjudge.cn/practice/24591/



思路：



代码

```python
def precedence(operator):
    if operator == '+' or operator == '-':
        return 1
    elif operator == '*' or operator == '/':
        return 2
    else:
        return 0
        
def infix_to_postfix(expression):
    stack = []
    result = []
    number = ''
    
    for char in expression:
        if char.isdigit() or char == '.':
            number += char
        elif number:
            result.append(number)
            number = ''
        
        if char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                result.append(stack.pop())
            stack.pop()  
        elif char in {'+', '-', '*', '/'}:
            while stack and precedence(stack[-1]) >= precedence(char):
                result.append(stack.pop())
            stack.append(char)
    
    if number:
        result.append(number)
    
    while stack:
        result.append(stack.pop())
    
    return ' '.join(result)

if __name__ == "__main__":
    n = int(input())
    for _ in range(n):
        expression = input().strip()
        print(infix_to_postfix(expression)) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-17 at 5.12.55 PM.png>)



### 22068: 合法出栈序列

http://cs101.openjudge.cn/practice/22068/



思路：



代码

```python
def ispopSeq(s1,s2):
    stack = []
    if len(s1) != len(s2):
        return False
    else:
        L = len(s1)
        stack.append(s1[0])
        p1,p2 = 1,0
        while p1 < L:
            if len(stack) > 0 and stack[-1] == s2[p2]:
                stack.pop()
                p2 += 1
            else:
                stack.append(s1[p1])
                p1 += 1
        return ''.join(stack[::-1]) == s2[p2:]
    
s1 = input()
while True:
    try:
        s2 = input()
    except:
        break
    if ispopSeq(s1,s2):
        print('YES')
    else:
        print('NO')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-13 at 2.03.48 PM.png>)



### 06646: 二叉树的深度

http://cs101.openjudge.cn/practice/06646/



思路：



代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))

def buildTree(nodes):
    if not nodes:
        return None
    tree_map = {}
    for i, (left, right) in enumerate(nodes):
        tree_map[i + 1] = TreeNode(i + 1)
    for i, (left, right) in enumerate(nodes):
        node = tree_map[i + 1]
        if left != -1:
            node.left = tree_map[left]
        if right != -1:
            node.right = tree_map[right]
    return tree_map[1]

if __name__ == "__main__":
    n = int(input())
    nodes = [tuple(map(int, input().split())) for _ in range(n)]
    root = buildTree(nodes)
    depth = maxDepth(root)
    print(depth)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-17 at 4.54.45 PM.png>)



### 02299: Ultra-QuickSort

http://cs101.openjudge.cn/practice/02299/



思路：



代码

```python
def merge_sort(a):
    if len(a) <= 1:
        return a,0

    mid = len(a)// 2
    left,left_count = merge_sort(a[:mid])
    right, right_count = merge_sort(a[mid:])

    merged, inv_merge = merge(left, right)

    return merged, left_count + right_count + inv_merge

def merge(left,right):
    merged = []
    counts = 0
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            counts += len(left) - i 
            
    merged += left[i:]
    merged += right[j:]

    return merged, counts

if __name__ == "__main__":
    while True:
        n = int(input())
        if n == 0:
            break
        a = [int(input()) for _ in range(n)]
        _, swaps = merge_sort(a)
        print(swaps) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-17 at 8.04.35 PM.png>)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

有点难，中序转后序那题来回修改了很多次才ac，二叉树和Ultra-QuickSort不会做，得看了gpt给的解释才大概明白。



