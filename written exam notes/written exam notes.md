# Written Exam Notes

## 时间复杂度
https://www.hello-algo.com/chapter_computational_complexity/time_complexity.assets/time_complexity_simple_example.png
*系数不影响阶数

**O(1) < O(logn) < O(n) < O(nlogn) < O(n^2) < O(2^n) < O(n!)**

O(1)常数阶
操作数量与输入数据大小n无关，即不随着n的变化而变化

0(n)线性阶
单层循环
遍历数组和遍历链表

O(n^2)平方阶
嵌套循环

O(2^n)指数阶
递归
https://www.hello-algo.com/chapter_computational_complexity/time_complexity.assets/time_complexity_exponential.png

O(logn)对数阶
和指数阶相反
每轮缩减到一半
递归
https://www.hello-algo.com/chapter_computational_complexity/time_complexity.assets/time_complexity_logarithmic.png

O(nlogn)线性对数阶
https://www.hello-algo.com/chapter_computational_complexity/time_complexity.assets/time_complexity_logarithmic_linear.png
快速排序、归并排序、堆排序
常出现于嵌套循环中，两层循环的时间复杂度分别为0(logn)和O(n)

O(n!)阶乘阶
递归
https://www.hello-algo.com/chapter_computational_complexity/time_complexity.assets/time_complexity_factorial.png

**常用数据结构的时间复杂度总结**

https://blog.csdn.net/lvlinfeng970/article/details/105383365

## 数据结构

### 逻辑结构
1. 线性结构：数组、链表、栈、队列、哈希表，元素之间是一对一的顺序关系。
2. 非线性结构：树、堆、图、哈希表。

### 存储结构
1. 连续：数组【栈、队列、哈希表、树、堆、图、矩阵】
2. 分散：链表【栈、队列、哈希表、树、堆、图】

### 数组
数组的长度是固定的

### 链表

#### **线性表**

线性表中数据元素之间的关系是一对一的关系，即除了第一个和最后一个数据元素之外，其它数据元素都是首尾相接的
说“线性”和“非线性”，只在逻辑层次上讨论，而不考虑存储层次，所以双向链表和循环链表依旧是线性表。
#除第一个和最后一个元素外，其余每个元素都有一个且仅有一个直接前驱和直接后继
#因为在单链表中插入元素需要遍历链表找到正确的位置，而每次插入的时间复杂度是O(n)，如果需要插入n个元素，那么总的时间复杂度就是O(n^2)

**双向链表**

![alt text](<Screenshot 2024-06-13 at 11.46.39 AM.png>)
例子：https://tsejx.github.io/data-structure-and-algorithms-guidebook/data-structure/linear-list/double-linked-list

**单链表**

存储密度 < 1

**有序表**

#插入新元素，平均移动n/2
#合并两个有序表，最少比较n次



## 排序算法

### 1.插入排序
**a.直接插入**
基本思想：顺序地把待排序的序列中各个元素按关键字的大小，插入到已排序的序列的适当位置
动图演示：https://www.runoob.com/wp-content/uploads/2019/03/insertionSort.gif

https://img-blog.csdn.net/20160316145805012

*原始序列**正序**排序时，性能最佳O(n)
        **逆序**排序时，排序效果最差O(n^2)
平均情况O(n^2)
空间复杂度O(1)
**稳定**

**b.Shell希尔排序**
效率较直接插入更好
基本思想：先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序，待整个序列中的记录"基本有序"时，再对全体记录进行依次直接插入排序。
具体解释：https://www.cnblogs.com/chengxiao/p/6104371.html
**不稳定**

### 2.选择排序
**a.直接选择**
数据规模越小越好
基本思想：首先在未排序的数列中找到最小(or最大)元素，然后将其存放到数列的起始位置；接着，再从剩余未排序的元素中继续寻找最小(or最大)元素，然后放到已排序序列的末尾。以此类推
动图演示：https://www.runoob.com/wp-content/uploads/2019/03/selectionSort.gif

最好最坏情况都是O(n^2)
空间复杂度O(1)
**不稳定**

**b.堆排序**
基本思想：将待排序序列构造成一个大顶堆，此时，整个序列的最大值就是堆顶的根节点。将其与末尾元素进行交换，此时末尾就为最大值。然后将剩余n-1个元素重新构造成一个堆，这样会得到n个元素的次小值。如此反复执行
具体解释：https://www.cnblogs.com/chengxiao/p/6129630.html

最好最坏情况都是O(nlogn)
空间复杂度O(1)
**不稳定**

### 3.交换排序
**a.冒泡排序**
基本思想：一次比较两个元素，如果他们的顺序错误就把他们交换过来
动图演示：https://www.runoob.com/wp-content/uploads/2019/03/bubbleSort.gif

最好情况O(n^2)
最坏情况O(n)
平均O(n^2)
空间复杂度O(1)
稳定

**b.快速排序**
基本思想：
1.选取数组最左端元素作为基准数
2.重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置
3.递归地把小于基准值元素的子数列和大于基准值元素的子数列排序
动图展示：https://www.runoob.com/wp-content/uploads/2019/03/quickSort.gif

最好情况O(n^2)
最坏情况O(nlog2n)
平均O(nlog2n)
空间复杂度O(nlog2n)
**不稳定**

### 4.归并排序
https://www.hello-algo.com/chapter_sorting/merge_sort.assets/merge_sort_overview.png


最好最坏情况都是O(nlog2n)
空间复杂度O(n)
**稳定**

### 5.桶排序
基本思想：初始化k个桶（每个桶有上下限），将待排序的元素放入这k个桶内，排序
https://www.hello-algo.com/chapter_sorting/bucket_sort.assets/bucket_sort_overview.png

时间复杂度O(n+k)
空间复杂度O(n+k)
**是否稳定取决于排序桶内元素的算法是否稳定**

### 6.计数排序
动图展示：https://www.runoob.com/wp-content/uploads/2019/03/countingSort.gif
*只适用于非负整数及数据范围较小的情况

时间复杂度O(n+m)
空间复杂度O(n+m)
**稳定**

### 7.基数排序
三者的差异：
基数排序：根据键值的每位数字来分配桶
计数排序：每个桶只存储单一键值
桶排序：每个桶存储一定范围的数值

动图展示：https://www.runoob.com/wp-content/uploads/2019/03/radixSort.gif

时间复杂度O(nk)、O(d(r+n))
空间复杂度O(rd+n)
**稳定**

## 树
节点数 = 度为0的节点数 + 度为1的节点数 + 度为2的节点数 + 度为3的节点数
度为0的节点数 = 度为2的节点数+1
叶节点 = 度为2的节点 + 1
边数 = 总节点数 − 1
深度 = 高度 - 1

## 图

节点 = 总度数 + 1

#### 1.拓扑结构（判断是否有环）

**1.1、无向图**
**使用拓扑排序可以判断一个无向图中是否存在环，具体步骤如下：**

1. 求出图中所有结点的度。
2. 将所有度 <= 1 的结点入队。（独立结点的度为 0）
3. 当队列不空时，弹出队首元素，把与队首元素相邻节点的度减一。如果相邻节点的度变为一，则将相邻结点入队。
4. 循环结束时判断已经访问的结点数是否等于n。等于n说明全部结点都被访问过，无环；反之，则有环。

**度 = 2 * 边**
   
**1.2、有向图**
**使用拓扑排序判断无向图和有向图中是否存在环的区别在于：**

在判断无向图中是否存在环时，是将所有度 <= 1 的结点入队；在判断有向图中是否存在环时，是将所有入度 = 0 的结点入队。

## 查找

#平衡因子是节点的左子树高度减去右子树高度，平衡因子为0表示左右子树高度相同。
#B-树是一种平衡树，所有叶子节点都在同一层次上。B-树的结构性排序属性是指节点内关键字的顺序性和节点之间的层级结构性。根节点的关键字集是有序的，但这是所有 B-树节点的普遍特性，不仅仅是根节点。

### 哈希表
哈希表容量越大，多个key被分配到同一个桶中的概率就越低，冲突就越少。因此，我们可以通过扩容哈希表来减少哈希冲突。
哈希表的结构改良方法主要包括“链式地址”和“开放寻址”。

链式地址：将单个元素转换为链表，将键值对作为链表节点，将所有发生冲突的键值对都存储在同一链表中。
在链地址法中，查找一个元素的时间取决于该位置链表的长度。
https://www.hello-algo.com/chapter_hashing/hash_collision.assets/hash_table_chaining.png

开放寻址：
1. 线性探测：通过哈希函数计算桶索引，若发现桶内已有元素，则从冲突位置向后线性遍历（步长通常为1），直至找到空桶，将元素插入其中。容易产生“聚集现象”
2. 平方探测
3. 多次哈希

## 错题

### 20240507

#### 选择题

4. 若某线性表常用操作是在表尾插入或删除元素，则时间开销最小的存储方式是（ C ）。 
   
A：单链表（从头遍历） B：仅有头指针的单循环链表（从头遍历） C：顺序表（可直接插入或删除） D：仅有尾指针的单循环链表（可直接插入，但要从头遍历才能删除）

11.  排序算法依赖于对元素序列的多趟比较/移动操作（即执行多轮循环），第一趟结束后，任一元素 都无法确定其最终排序位置的算法是（ D ）。
   
A：选择排序 B：快速排序 C: 冒泡排序 D：插入排序（只能保证当前元素在已排序部分的正确位置）

12.  考察以下基于单链表的操作，相较于顺序表实现，带来更高时间复杂度的操作是（ D ）。 

A：合并两个有序线性表，并保持合成后的线性表依然有序
13.  已知一个整型数组序列，序列元素值依次为 ( 19，20，50，61，73，85，11，39 )，采用某种排序算法，在多趟比较/移动操作（即执行多轮循环）后，依次得到以下中间结果（每一行对应一趟）如下： （1）19 20 11 39 73 85 50 61 （2）11 20 19 39 50 61 73 85 （3）11 19 20 39 50 61 73 85 请问，上述过程使用的排序算法是（ C ）。

A：冒泡排序 B：插入排序 C：希尔排序（分别取间隔4、2、1） D：归并排序

14. 今有一非连通无向图，共有 36 条边，该图至少有（ C ）个顶点。 

A：8 B：9 C：10 D：11
一个连通分量的边数最多为 [n(n-1)]/2 (n是顶点数)
会算到9，但因为是非联通图，所以要+1

### 判断题

2. （N）构建一个含 N 个结点的（二叉）最小值堆，时间效率最优情况下的时间复杂度大O表示为O(NLogN)。 （O(NlogN)是最差情况，最优情况应是O(N)
3. （N）对任意一个连通的无向图，如果存在一个环，且这个环中的一条边的权值不小于该环中任意一个其它的边的权值，那么这条边一定不会是该无向图的最小生成树中的边。 
   [最小生成树是一个连通无向图的一个子图，它包含了图中的所有顶点，并且其总边权值最小，同时没有环。]
5. （ Y ）树可以等价转化二叉树，树的先序遍历序列与其相应的二叉树的前序遍历序列相同。

### 填空题

1. 定义二叉树中一个结点的度数为其子结点的个数。现有一棵结点总数为101的二叉树，其中度数为1的结点数有30个，则度数为0结点有 _36_ 个。
   
2. 对于初始排序码序列（51,41,31,21,61,71,81,11,91），用双指针原地交换实现，第1趟快速排序（以第一个数字为中值）的结果是： _11 41 31 21 51 71 81 61 91_ 。
   
#需要定义两个指针，一个指向序列的开头，另一个指向序列的末尾。
#从序列的末尾开始，找到第一个小于中值51的数字，即11。
#从序列的开头开始，找到第一个大于中值51的数字，即61。
#交换数字

1. 已知某二叉树的先根周游序列为 ( A,B,D,E,C,F,G)，中根周游序列为 (D,B,E,A,C,G,F)，则该二叉树的后根次序周游序列( _D,E,B,G,F,C,A_ )。
   
#F是G的右节点，G是C的右节点

1. 51个顶点的连通图G有50条边，其中权值为1,2,3,4,5,6,7,8,9,10的边各5条，则连通图G的最小生成树各边的权值之和为 _275_ 。
   
#最小生成树的边数等于顶点数-1
#1*5 + 2*5 + 3*5 + 4*5 + 5*5 + 6*5 + 7*5 +8*5 + 9*5 + 10*5
#= 5+10+15+20+25+30+35+40+45+50
#= 275

1. 包含n个顶点无向图的邻接表存储结构中，所有顶点的边表中最多有 _n(n-1)_ 个结点。具有n个顶点的有向图，顶点入度和出度之和最大值不超过 _2(n-1)_ 。
   
#最多有n−1条入边和n−1条出边

1. 给定一个长度为7的空散列表，采用双散列法解决冲突，两个散列函数分别为：h1(key) = key % 7，h2(key) = key%5 + 1 请向散列表依次插入关键字为 30, 58, 65 的集合元素，插入完成后 65 在散列表中存储地址为 _3_ 。
   
#因为关键字65的初始散列地址与关键字30的初始散列地址相同，我们需要使用第二个散列函数计算增量，直到找到一个空的槽位。根据第二个散列函数，计算增量为1。因此，我们需要将关键字65插入到初始散列地址为2+1.

### 20240514

#### 选择题

2. 给定一个 N 个相异元素构成的有序数列，设计一个递归算法实现数列的二分查找，考察递归过程中栈的使用情况，请问这样一个递归调用栈的最小容量应为（ D ）。 
   
A：N B：N/2 C：$\lceil \log_{2}(N) \rceil$ D：$\lceil \log_{2}(N+1) \rceil$​

#考虑边界情况

4. 为了实现一个循环队列（或称环形队列），采用数组 Q[0..m-1]作为存储结构,其中变量rear表示这个循环队列中队尾元素的实际位置，添加结点时按rear=(rear+1) % m进行指针移动，变量length表示当前队列中的元素个数，请问这个循环队列的队列首位元素的实际位置是（ B ）。 

A：rear-length B：(1+rear+m-length) % m C：(rear-length+m) % m D：m-length

#length = rear - head + 1，再对环形队列的特点做调整，得到B。

8. 已知一个无向图G含有18条边，其中度数为4的顶点个数为3，度数为3的顶点个数为4，其他顶点的度数均小于3，请问图G所含的顶点个数至少是（ C ）。 
   
A: 10 B: 11 C: 13 D: 15

#一条边贡献两个度数

14. 考虑一个森林F，其中每个结点的子结点个数均不超过2。如果森林F中叶子结点的总个数为L，度数为2结点（子结点个数为2）的总个数为N，那么当前森林F中树的个数为（ C ）。 
A：L-N-1 B：无法确定 C：L-N D：N-L

#一棵树中，叶节点比二度节点数量多一

### 填空题

2. 在一棵含有n个结点的树中，只有度（树节点的度指子节点数量）为k的分支结点和度为0的终端（叶子）结点，则该树中含有的终端（叶子）结点的数目为： _n - (n-1)/k_ 。

#设叶节点x个，分支节点n-x个，树的性质有节点数等于所有节点度数和+1，(n-x)*k + 1 = n, x = n - (n-1)/k

4. 对长度为3的顺序表进行查找，若查找第一个元素的概率为1/2，查找第二个元素的概率为1/4，查找第三个元素的概率为1/8，则执行任意查找需要比较元素的平均个数为 _1.75_ 。

#$1*(1/2) + 2*(1/4) + 3*(1/8) + 3*(1/8) = 1.75$, 还有1/8的失败查询概率。

7. 已知以数组表示的小根堆为[8，15，10，21，34，16，12]，删除关键字8之后需要重新建堆，在此过程中，关键字的比较次数是 _3_ 。
   