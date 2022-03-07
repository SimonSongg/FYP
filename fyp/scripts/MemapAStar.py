#!/usr/bin/env python
# coding: utf-8





import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Memap:
    
    def __init__(self,pointCloudPATH):
        self.pointCloud=np.array(pd.read_excel(pointCloudPATH))
        #自动构建地图
        #地图数据结构声明
        # 本质上是一串链表
        # 就是链表组成的列表
        self.memap=[[(1,2),[(1,(5,5)),(2,(2,3))]],
                    [(5,5),[(0,(1,2)),(2,(2,3))]],
                    [(2,3),[(0,(1,2)),(1,(5,5))]]]
        # for node in memap:
        #     #node是地图的基本结构，内包含此节点所有信息
        #     #第一位是node本身的坐标
        #     print('node is:'+str(node[0]))
        #     for ixy in node[-1]:
        #         #ixy第一位是此节点可以通向的节点的索引
        #         #ixy后一位是此节点可以通向的节点坐标
        #         print('nodes index is:'+str(ixy[0])+'    Adjacent nodes is:'+str(ixy[-1]))
        #初始化地图
        self.memap.clear()
        #初始化地图标记
        self.pointCloudSIGN=np.zeros(len(self.pointCloud))
        for i in range(len(self.pointCloud)):
            minDistance=np.Inf
            a_point = self.pointCloud[i]
            for j in range(len(self.pointCloud)):
                b_point = self.pointCloud[j]
                EuclideanDistance=self.EuclideanDistanceCostFun(a_point,b_point)
                #求取最小值，不能是本身，也不能是已经用过的node
                if i!=j and self.pointCloudSIGN[j]==0 and EuclideanDistance<minDistance:
                    minDistance=EuclideanDistance
                    minCostNode=b_point
                    minNodeIndex=j
            self.pointCloudSIGN[i]=1
            self.memap.append([tuple(a_point),[(minNodeIndex,tuple(minCostNode))]])

        for i in range(len(self.memap)):
            nodeXY,nextNode = self.memap[i]
            self.memap[nextNode[0][0]][-1].append((i,nodeXY))
            
            
    # 欧式距离代价函数
    def EuclideanDistanceCostFun(self,a,b):
        x=a[0]-b[0]
        y=a[1]-b[1]
        return math.sqrt(x*x+y*y)

    
    def showMap(self):
        for node in self.memap:
            for nextNode in node[-1]:
                route_lim = np.array((node[0],nextNode[-1]))
                plt.plot(route_lim[:,0],route_lim[:,1],'r')
        plt.show()
        
        
    # 手动添加缺失路线的函数
    def addRoute(self,a,b):
        flag1=flag2=False
        for i in range(len(self.memap)):
            node=self.memap[i]
            nodeXY,nextNode=node
            if a==nodeXY:
                a_index=i
                flag1=True
            if b==nodeXY:
                b_index=i
                flag2=True
            if flag1 and flag2:
                break
        self.memap[a_index][-1].append((b_index,b))
        self.memap[b_index][-1].append((a_index,a))
        
        
    def delRoute(self,a,b):
        flag1=flag2=False
        for i in range(len(self.memap)):
            node=self.memap[i]
            nodeXY,nextNode=node
            if a==nodeXY:
                a_index=i
                flag1=True
            if b==nodeXY:
                b_index=i
                flag2=True
            if flag1 and flag2:
                break
        self.memap[a_index][-1].remove((b_index,b))
        self.memap[b_index][-1].remove((a_index,a))
    
        
    def aStar(self,be,en):
        # 声明一个地图标记用于存储走过的信息
        self.began=be
        self.end=en
        memap_ = np.zeros(len(self.memap))
        open_set = set()
        close_set = set()
        old_open_set = ()
        be=tuple(be)
        en=tuple(en)

        for i in range(len(self.memap)):
            if self.memap[i][0]==be:
                open_set.add(be+(0,self.EuclideanDistanceCostFun(be,en))+(i,-1))
        #setItem说明：（1,2,3,4,5,6）
        #             1,2 存储当前node坐标
        #             3,4 存储H F
        #             5 存储本node的index
        #             6 存储上node的index
        while True:
            new_open_set = []
            min_cost=np.Inf
            min_node=()
            min_index_me=0
            min_index_last=0

            for open_item in open_set:
                node_x,node_y,cost_H,cost_F,indexMe,indexLast=open_item
                # memap_[indexMe] == 0 说明indexMe这个位置还没有走过
                if cost_H+cost_F<min_cost and memap_[indexMe]==0:
                    min_cost=cost_H+cost_F
                    min_node=(node_x,node_y)
                    min_index_me=indexMe
                    min_index_last=indexLast
                    old_open_set=open_item

            # 走过的位置掷为 1
            memap_[min_index_me] = 1
            if min_node==en:
                close_set.update(open_set)
                self.route_set = close_set
                return "Success"

            for nextNode in self.memap[min_index_me][-1]:
                nextIndex,nextXY=nextNode
                if memap_[nextIndex]==0:
                    cost_H+=self.EuclideanDistanceCostFun(min_node,nextXY)
                    cost_F+=self.EuclideanDistanceCostFun(nextXY,en)
                    new_open_set.append(nextXY+(cost_H,cost_F)+(nextIndex,min_index_me))

            close_set.update(open_set)
            if old_open_set in open_set:
                open_set.remove(old_open_set)
            open_set.update(tuple(new_open_set))
            
            
    # 路径回寻
    def findRoute(self):
        route_list=[]
        end_node=self.end
        while end_node!=self.began:
    #         print(end_node)
            for ii in self.route_set:
                if end_node == ii[0:2]:
                    route_list.append(end_node)
                    end_node = self.memap[ii[-1]][0]
                    break
        route_list.append(self.began)
        return route_list[::-1]
    

    # 路径绘制
    def show_route(self):
        route_list=np.array(self.findRoute())
        for node in self.memap:
            for nextNode in node[-1]:
                route_lim = np.array((node[0],nextNode[-1]))
                plt.plot(route_lim[:,0],route_lim[:,1],'k')
        plt.plot(self.pointCloud[:,0],self.pointCloud[:,1],'.k')
        plt.plot(route_list[:,0],route_list[:,1],'r')
        plt.plot(self.began[0],self.began[1],'*g')
        plt.plot(self.end[0],self.end[1],'*r')
        plt.show()
        plt.show()

