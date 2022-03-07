#!/usr/bin/env python
# coding: utf-8

#导入MemapAStar库
import MemapAStar


#用Memap方法构造一个地图，参数为GPS点云的.xlsx文件
#返回一个Memap对象
memap = MemapAStar.Memap(r'pointcloud.xlsx')
#当然你也可以使用input来自定义一个点云文件
#pointCloud = input('请输入.xlsx文件的位置：')
#memap = MemapAStar.Memap(pointCloud)


#addRoute方法接收两个参数
#用于在两个点之间创建一条路径（人工补齐地图缺损的位置）
#无返回值
memap.addRoute((2,6),(3,6))
memap.addRoute((2,4),(3,4))
memap.addRoute((5,4),(6,4))


#aStar方法接收两个参数
#寻找到这两个点的相对最优路径
#无返回值
memap.aStar((1,1),(7,9))


#findRoute方法没有参数
#用于回寻路径点集合，生成路径
#返回相对最优路径
for node in memap.findRoute():
    print(node)


#showMap方法用于展示自动构建的地图
#无参数
#无返回值
memap.showMap()


#show_route方法用于显示相对最优路径
#无参数
#无返回值
memap.show_route()



