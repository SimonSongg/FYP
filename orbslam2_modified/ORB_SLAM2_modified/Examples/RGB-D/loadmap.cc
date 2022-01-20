#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>


using namespace std;

int main(int argc, char **argv)
{
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);
    SLAM.LoadMap("/home/simon/map.bin");
}