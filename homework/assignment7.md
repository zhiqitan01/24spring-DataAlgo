# Assignment #7: April 月考

Updated 1557 GMT+8 Apr 3, 2024

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

### 27706: 逐词倒放

http://cs101.openjudge.cn/practice/27706/



思路：



代码

```python
a = input().split()
reverse_a = a[::-1]
print(' '.join(reverse_a))

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-05 at 10.14.10 AM.png>)



### 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/



思路：



代码

```python
from collections import deque

m,n = map(int,input().split())
a = list(map(int,input().split()))
queue = deque()
count = 0

for i in a:
    if i not in queue:
        count += 1
        queue.append(i)
        
    if len(queue) > m:
        queue.popleft()
        
print(count)

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-05 at 10.15.32 AM.png>)



### 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/



思路：



代码

```python
def count_elements_less_or_equal(a, x):
    count = 0
    for i in a:
        if i <= x:
            count += 1
    return count

def find_min_x(a, k):
    a.sort()
    left = 1
    right = 10**9
    result = -1

    while left <= right:
        mid = (left + right) // 2
        if count_elements_less_or_equal(a, mid) == k:
            result = mid
            right = mid - 1
        elif count_elements_less_or_equal(a, mid) < k:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

n, k = map(int, input().split())
a = list(map(int, input().split()))

min_x = find_min_x(a, k)
print(min_x)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-08 at 1.16.47 PM.png>)



### 27948: FBI树

http://cs101.openjudge.cn/practice/27948/



思路：



代码

```python
def build_FBI_tree(N, s):
    if len(s) == 1:  
        if s == '0':
            return 'B'
        else:
            return 'I'
    else:
        mid = len(s) // 2
        left = s[:mid]  
        right = s[mid:]
        root_type = 'B' if '1' not in s else 'F' if '0' in s else 'I'  
        left_tree = build_FBI_tree(N - 1, left)  
        right_tree = build_FBI_tree(N - 1, right)  
        return left_tree + right_tree + root_type

N = int(input())
binary_string = input()
post_order_traversal = build_FBI_tree(N, binary_string)
print(post_order_traversal)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-08 at 1.22.23 PM.png>)



### 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/



思路：



代码

```python
from collections import deque					

t = int(input())
groups = {}
member_to_group = {}

for _ in range(t):
    members = list(map(int, input().split()))
    group_id = members[0]  
    groups[group_id] = deque()
    for member in members:
        member_to_group[member] = group_id

queue = deque()
queue_set = set()


while True:
    command = input().split()
    if command[0] == 'STOP':
        break
    elif command[0] == 'ENQUEUE':
        x = int(command[1])
        group = member_to_group.get(x, None)
        if group is None:
            group = x
            groups[group] = deque([x])
            member_to_group[x] = group
        else:
            groups[group].append(x)
        if group not in queue_set:
            queue.append(group)
            queue_set.add(group)
    elif command[0] == 'DEQUEUE':
        if queue:
            group = queue[0]
            x = groups[group].popleft()
            print(x)
            if not groups[group]: 
                queue.popleft()
                queue_set.remove(group)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-08 at 1.46.59 PM.png>)



### 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/



思路：



代码

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []


def traverse_print(root, nodes):
    if root.children == []:
        print(root.value)
        return
    pac = {root.value: root}
    for child in root.children:
        pac[child] = nodes[child]
    for value in sorted(pac.keys()):
        if value in root.children:
            traverse_print(pac[value], nodes)
        else:
            print(root.value)


n = int(input())
nodes = {}
children_list = []
for i in range(n):
    info = list(map(int, input().split()))
    nodes[info[0]] = TreeNode(info[0])
    for child_value in info[1:]:
        nodes[info[0]].children.append(child_value)
        children_list.append(child_value)
root = nodes[[value for value in nodes.keys() if value not in children_list][0]]
traverse_print(root, nodes)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-04-08 at 1.48.26 PM.png>)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

比起上两次作业简单很多。月考时间我只来得及ac2题，medium级别题目也能自己完成，但花较长时间。



