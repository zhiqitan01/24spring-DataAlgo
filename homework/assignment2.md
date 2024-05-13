# Assignment #2: 编程练习

Updated 0953 GMT+8 Feb 24, 2024

2024 spring, Complied by 陈紫琪，信息管理系



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS Ventura 13.4.1 (c)

Python编程环境：Spyder IDE 5.2.2



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：在计算分母那边卡了挺久的，不知道要怎么写才对，结果就误打误撞ac了



##### 代码

```python
def solve(a,b):  #计算最大公约数
    while b:
        a,b = b, a % b
    return a

def fraction(n):
    numerator = n[0] * n[3] + n[2] * n[1] #计算分子
    denominator = n[1] * n[3] #计算分母
    divisor = solve(numerator,denominator)
    result_numerator = numerator // divisor
    result_denominator = denominator // divisor
    return str(result_numerator) + "/" + str(result_denominator)
    
n = list(map(int,input().split()))
print(fraction(n))

```


代码运行截图 ==（至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.22.01 PM-1.png>)



### 04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110



思路：



##### 代码

```python
n,w = map(int,input().split())
candy = []
for _ in range(n):
    v,weight = map(int,input().split())
    candy.append((v/weight,v,weight))
    
candy.sort(reverse = True)
total_value = 0
for i,v,weight in candy:
    if w >= weight:
        total_value += v
        w -= weight
    else:
        total_value += w*i
        break
    
print("{:.1f}".format(total_value))

```



代码运行截图 ==（至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.38.16 PM.png>)



### 18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/



思路：



##### 代码

```python
s = int(input())
for _ in range(s):
    n,m,b = map(int,input().split(" "))   #n=技能，m=每个时刻最多使用的技能，b=血量
    d = {}
    for _ in range(n):
        t,x = map(int,input().split(" "))   #t=时刻，x=血量下降
        if t not in d.keys():   #如果字典中没有记录过这一时刻
            d[t] = [x]   #将x添加到对应的列表中
        else:
            d[t].append(x)   #比如同一时刻有两种技能，{1:[5,5]} -- t:d[t]=[x]
            
    for i in d.keys():   #遍历每个时刻
        d[i].sort(reverse = True)   #对每个时刻里的技能按血量下降，降序排列
        d[i] = sum(d[i][:m])   #取排序后的前m个技能
    dp = sorted(d.items())   #对{时刻:血量下降}按照时刻顺序排列
    
    for i in dp:   #遍历每个{时刻：血量下降}
        b -= i[1]   #怪兽血量减去i[1]（血量下降）
        if b <= 0:
            print(i[0])   #print时刻
            break
    else:   #如果循环结束时怪兽血量仍然大于0
            print("alive")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.40.13 PM.png>)



### 230B. T-primes

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B



思路：这题和2050那题几乎是一样的，但是我用了不同的方式来解。



##### 代码

```python
def sieve_of_eratosthenes(n):
    prime = [True] * (n+1)
    prime[0] = prime[1] = False
    p = 2
    while p*p <= n:
        if prime[p] == True:
            for i in range(p*p, n+1, p): 
                prime[i] = False
        p += 1
    return [i for i in range(2, n+1) if prime[i]]
 
def is_t_prime(a):
    result = []
    max_num = max(a)
    primes = set(sieve_of_eratosthenes(int(max_num**0.5)+1))
    for num in a:
        if (num**0.5).is_integer() and int(num**0.5) in primes: 
            result.append("YES")
        else:
            result.append("NO")
    return result
    
n = int(input())
a = list(map(int,input().split()))
for i in is_t_prime(a):
    print(i)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.41.45 PM.png>)



### 1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A



思路：实在想不到怎么做，参考了github里的答案写出来的



##### 代码

```python
def prefix_sum(nums):
    prefix = []
    total = 0
    for i in nums:
        total += i
        prefix.append(total)
    return prefix
 
def suffix_sum(nums):
    suffix = []
    total = 0
    reversed_nums = nums[::-1]
    for i in reversed_nums:
        total += i
        suffix.append(total)
    suffix.reverse()
    return suffix
 
 
t = int(input())
for _ in range(t):
    N, x = map(int, input().split())
    a = [int(i) for i in input().split()]
    aprefix_sum = prefix_sum(a)
    asuffix_sum = suffix_sum(a)
 
    left = 0
    right = N - 1
    if right == 0:
        if a[0] % x !=0:
            print(1)
        else:
            print(-1)
        continue
 
    leftmax = 0
    rightmax = 0
    while left != right:
        total = asuffix_sum[left]
        if total % x != 0:
            leftmax = right - left + 1
            break
        else:
            left += 1
 
    left = 0
    right = N - 1
    while left != right:
        total = aprefix_sum[right]
        if total % x != 0:
            rightmax = right - left + 1
            break
        else:
            right -= 1
    
    if leftmax == 0 and rightmax == 0:
        print(-1)
    else:
        print(max(leftmax, rightmax))


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-03 at 3.51.55 PM.png>)



### 18176: 2050年成绩计算

http://cs101.openjudge.cn/practice/18176/



思路：这题的解法会比tprime那题更简单一点



##### 代码

```python
m,n=map(int,input().split())
t=[True]*(10001)

for x in range(0,10001):
    t[0]=False
    t[1]=False
    t[2]=True
    if t[x]:
        for y in range(x*2,10000,x):
            t[y]=False
            
def t_prime(score):
    if t[int(score**0.5)]==True and int(score**0.5)==score**0.5:
        return True
    return False
            
            
for i in range(m):
    scores=list(map(int,input().split()))
    valid_scores=[score for score in scores if t_prime(score)]
    
    if len(valid_scores)==0:
        print(0)
    else:
        print("{:.2f}".format(sum(valid_scores)/ len(scores)))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![Alt text](<Screenshot 2024-03-02 at 5.43.44 PM.png>)



## 2. 学习总结和收获

这次的作业也是有几题在以前的计概中做过，整体来说还算可以，但也有不会做的和花了很长时间才做出来的。




