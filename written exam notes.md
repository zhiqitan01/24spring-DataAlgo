## Written Exam Notes

### 排序算法
**1.插入排序**
a.直接插入
基本思想：顺序地把待排序的序列中各个元素按关键字的大小，插入到已排序的序列的适当位置
动图演示：https://www.runoob.com/wp-content/uploads/2019/03/insertionSort.gif

https://img-blog.csdn.net/20160316145805012

*原始序列**正序**排序时，性能最佳O(n)
        **逆序**排序时，排序效果最差O(n^2)
平均情况O(n^2)
空间复杂度O(1)
稳定

b.Shell希尔排序
效率较直接插入更好
基本思想：先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序，待整个序列中的记录"基本有序"时，再对全体记录进行依次直接插入排序。
具体解释：https://www.cnblogs.com/chengxiao/p/6104371.html
不稳定

**2.选择排序**
a.直接选择
数据规模越小越好
基本思想：首先在未排序的数列中找到最小(or最大)元素，然后将其存放到数列的起始位置；接着，再从剩余未排序的元素中继续寻找最小(or最大)元素，然后放到已排序序列的末尾。以此类推
动图演示：https://www.runoob.com/wp-content/uploads/2019/03/selectionSort.gif

最好最坏情况都是O(n^2)
空间复杂度O(1)
不稳定

b.堆排序
基本思想：将待排序序列构造成一个大顶堆，此时，整个序列的最大值就是堆顶的根节点。将其与末尾元素进行交换，此时末尾就为最大值。然后将剩余n-1个元素重新构造成一个堆，这样会得到n个元素的次小值。如此反复执行
具体解释：https://www.cnblogs.com/chengxiao/p/6129630.html

最好最坏情况都是O(nlogn)
空间复杂度O(1)
不稳定

**3.交换排序**
a.冒泡排序
基本思想：一次比较两个元素，如果他们的顺序错误就把他们交换过来
动图演示：https://www.runoob.com/wp-content/uploads/2019/03/bubbleSort.gif

最好情况O(n^2)
最坏情况O(n)
平均O(n^2)
空间复杂度O(1)
稳定

b.快速排序
基本思想：
1.选取数组最左端元素作为基准数
2.重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置
3.递归地把小于基准值元素的子数列和大于基准值元素的子数列排序
动图展示：https://www.runoob.com/wp-content/uploads/2019/03/quickSort.gif

最好情况O(n^2)
最坏情况O(nlog2n)
平均O(nlog2n)
空间复杂度O(nlog2n)
不稳定

**4.归并排序**
https://www.hello-algo.com/chapter_sorting/merge_sort.assets/merge_sort_overview.png


最好最坏情况都是O(nlog2n)
空间复杂度O(n)
稳定

**5.桶排序**
基本思想：初始化k个桶（每个桶有上下限），将待排序的元素放入这k个桶内，排序
https://www.hello-algo.com/chapter_sorting/bucket_sort.assets/bucket_sort_overview.png

时间复杂度O(n+k)
空间复杂度O(n+k)
是否稳定取决于排序桶内元素的算法是否稳定

**6.计数排序**
动图展示：https://www.runoob.com/wp-content/uploads/2019/03/countingSort.gif
*只适用于非负整数及数据范围较小的情况

时间复杂度O(n+m)
空间复杂度O(n+m)
稳定

**7.基数排序**
三者的差异：
基数排序：根据键值的每位数字来分配桶
计数排序：每个桶只存储单一键值
桶排序：每个桶存储一定范围的数值

动图展示：https://www.runoob.com/wp-content/uploads/2019/03/radixSort.gif

时间复杂度O(nk)、O(d(r+n))
空间复杂度O(rd+n)
稳定

### 时间复杂度
https://www.hello-algo.com/chapter_computational_complexity/time_complexity.assets/time_complexity_simple_example.png
*系数不影响阶数

O(1)<O(logn)<O(n)<O(nlogn)<O(n^2)<O(2^n)<O(n!)

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
