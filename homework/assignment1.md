# Assignment #1: 拉齐大家Python水平

Updated 0940 GMT+8 Feb 19, 2024

2024 spring, Complied by 陈紫琪，信息管理系



**说明：**

1）数算课程的先修课是计概，由于计概学习中可能使用了不同的编程语言，而数算课程要求Python语言，因此第一周作业练习Python编程。如果有同学坚持使用C/C++，也可以，但是建议也要会Python语言。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS Ventura 13.4.1 (c)

Python编程环境：Spyder IDE 5.2.2





## 1. 题目

### 20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/



思路：和fibonacci数列做法很相似



##### 代码

```python
def function(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return function(n-3) + function(n-2) + function(n-1)
    
n = int(input())
print(function(n))
```



代码运行截图 ==（至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.12.34 PM.png>)



### 58A. Chat room

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A



思路：



##### 代码

```python
def check(s):
    hello = "hello"
    i = 0
    j = 0
    for i in range(len(s)):
        if j < len(hello) and s[i] == hello[j]:
            j += 1
    return j == len(hello)
 
s = input()
if check(s):
    print("YES")
else:
    print("NO")

```



代码运行截图 ==（至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.17.16 PM.png>)



### 118A. String Task

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A



思路：



##### 代码

```python
vowels = ["a","e","i","o","u","y"]
a = input().lower()
result = ""
for i in a:
    if i not in vowels:
        result += "." + i
        
print(result)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.19.05 PM.png>)



### 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/



思路：用筛法来做



##### 代码

```python
def sieve_of_eratosthenes(n,prime):
    prime[0] = prime[1] = False
    for i in range(2,n+1):
        prime[i] = True
        
    p = 2
    while p*p <= n:
        if prime[p] == True:
            i = p*p
            while i <= n:
                prime[i] = False
                i += p
                
        p += 1
        
def isPrime(n):
    prime = [0]*(n+1)
    sieve_of_eratosthenes(n,prime)
    for i in range(n):
    	if prime[i] and prime[n - i]: 
    		print(i,(n - i)) 
    		return
        
n = int(input())
isPrime(n)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.19.58 PM.png>)



### 23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/



思路：



##### 代码

```python
n = list(input().split("+"))
a = []
result = 0
for i in n:
    b,c = i.split("n^")
    a.append((b,c))
    
for i in a:
    if i[0] != "0" and int(i[1]) > result:
        result = int(i[1])
        
print("n^{}".format(result)) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.21.00 PM.png>)



### 24684: 直播计票

http://cs101.openjudge.cn/practice/24684/



思路：用字典的方式来做



##### 代码

```python
votes = list(map(int,input().split()))
count = {}
for i in votes:
    count[i] = 	count.get(i,0)+1
    
    max_votes = max(count.values())
    result = [i for i,votes in count.items() if votes == max_votes]
    
result.sort()
print(*result,sep = " ") 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.22.01 PM.png>)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“数算pre每日选做”、CF、LeetCode、洛谷等网站题目。==

这次的题目中大部分是之前计概做过的题，相对来说比较简单。除了最后一题一因为不太熟悉字典的语法所以用了gpt外，其他都能独立完成。



