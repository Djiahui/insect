# insects generating model
In this model, the number of insects to be generated 
is determined by current number and temperature. The 
temperature is directly from data in that tex file.
# predict model
仓库中机器的位置上粮食比较多，有一定几率虫子会被卷入到粮食中，其他位置的虫子不会（小概率）被卷入粮食中。是否被卷入到粮食中跟在某一个区域待得时间长短有关，成正相关。

每一个位置上每一天粮食会有一个匀速的上升

虫子的移动仍然跟粮食的数量有关,不和每只虫子的

最终的损失和被抓走的虫子有关 

四个黑色方块、圆圈、长方形，其余空白，各设置一种food（ij）

先计算虫子被捕捉的概率，再计算虫子被捕捉的数量，然后得到一个概率