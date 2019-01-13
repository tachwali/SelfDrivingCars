#pragma once
#include <vector>

struct carStates
    {
        bool car_ahead;
        bool car_left;
        bool car_right;
    } ;


struct carActions
{
    double speed_diff;
    int lane_change;
};


struct splineAnchors
{
    std::vector<double> ptsx;
    std::vector<double> ptsy;
};