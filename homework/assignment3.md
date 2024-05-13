# Assignment #3: March月考

Updated 1537 GMT+8 March 6, 2024

2024 spring, Complied by 陈紫琪，信息管理系



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS Ventura 13.4.1 (c)

Python编程环境：Spyder IDE 5.2.2



## 1. 题目

**02945: 拦截导弹**

http://cs101.openjudge.cn/practice/02945/



思路：一开始没想那么多，以为只是简单的统计高度小于前一发导弹的数量，结果就wa了。之后重新看过几遍题目后才发现要用dp来做



##### 代码

```python
def bomb(height):
    dp = [1] * len(height)  
    
    for i in range(1, len(height)):
        for j in range(i):
            if height[i] <= height[j]:
                dp[i] = max(dp[i], dp[j] + 1)  
    
    return max(dp)  

k = int(input())
height = list(map(int, input().split()))

print(bomb(height))

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-06 at 5.06.31 PM.png>)



**04147:汉诺塔问题(Tower of Hanoi)**

http://cs101.openjudge.cn/practice/04147



思路：我觉得这题不太easy😭，可能是还不熟练递归



##### 代码

```python
def move_disk(n, a, c):
    print(f"{n}:{a}->{c}")

def hanoi_tower(n,a,b,c):
    if n == 1:
        move_disk(n,a,c)
    else:
        hanoi_tower(n - 1,a,c,b)
        move_disk(n,a,c)
        hanoi_tower(n - 1,b,a,c)

n, a, b, c = input().split()

hanoi_tower(int(n),a,b,c)

```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-06 at 5.07.42 PM.png>)



**03253: 约瑟夫问题No.2**

http://cs101.openjudge.cn/practice/03253



思路：正好才刚做了约瑟夫问题，稍微改了点就ac了



##### 代码

```python
def kids(n, p, m):
    children = list(range(1, n + 1)) 
    out = [] 
    current_index = p - 1  

    while children:
        current_index = (current_index + m - 1) % len(children)  
        out.append(children.pop(current_index))  

    return out

while True:
    n, p, m = map(int, input().split())
    if n == 0 and p == 0 and m == 0:
        break
    print(','.join(map(str, kids(n, p, m)))) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-06 at 5.10.35 PM.png>)



**21554:排队做实验 (greedy)v0.2**

http://cs101.openjudge.cn/practice/21554



思路：学到了enumerate的用法！



##### 代码

```python
n = int(input())
t = [(i,int(j)) for i,j in enumerate(input().split(),1)] #enumerate(iterable,start)
ans = []

s = t.copy()
s.sort(key = lambda x: x[1]) #x[1]对第二个元素进行排序
for i in s:
    ans.append(i[0])
    
print(*ans)

dp = [0]*n
dp[0] = 0
for i in range(1,n):
    dp[i] = dp[i-1] + s[i-1][1] #将前一个学生的时长总和dp[i-1]+当前学生的时长s[i-1][1]，然后将结果存储在dp[i]中
    
print("{:.2f}".format(sum(dp)/n))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-06 at 5.41.25 PM.png>)



**19963:买学区房**

http://cs101.openjudge.cn/practice/19963



思路：一开始不知道eval()，在input部分就卡了很久



##### 代码

```python
n = int(input())
distances = [eval(x)[0]+eval(x)[1] for x in input().split()] #eval将字符转成数值型用于计算
prices = [int(x) for x in input().split()]
value = [distances[x]/prices[x] for x in range(n)]

def mid(n,m):
    m = sorted(m)
    if n % 2 == 1:
        return m[n//2]
    else:
        return (m[n//2-1]+m[n//2])/2
    
h = mid(n,value) #性价比中位数
p = mid(n,prices) #价格中位数
sum = 0
for i in range(n):
    if value[i] > h and prices[i] < p:
        sum += 1
print(sum)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-06 at 6.07.39 PM.png>)



**27300: 模型整理**

http://cs101.openjudge.cn/practice/27300



思路：还记得这题是上次计概期末的留学生题。因为很多细节方面的小错误导致一直不能ac😭



##### 代码

```python
def f(a) :
    if a[-1]=='M':
        return(float(a[:-1]))
    elif a[-1]=='B':
        return(float(a[:-1])*1000)

n=int(input())
a=set()
b=dict()

for i in range(n):
    x=input().split('-')
    if x[0] in a:
        b[x[0]].append(x[1])
    else:
        a.add(x[0])
        b[x[0]]=[x[1]]
        
a=list(a)
a.sort()
for i in a:
    b[i].sort(key=f)
    t=', '.join(b[i])
    print(i+': '+t)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](<Screenshot 2024-03-07 at 7.48.30 PM.png>)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

这次花了点时间看了一点教材，感觉比直接刷题好多了，虽然还是不能很顺利的解题，但是头脑里有了更多思路。




