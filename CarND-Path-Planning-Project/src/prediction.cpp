#include "prediction.hpp"
#include <math.h>


carStates predict(std::vector<std::vector<float>> sensor_fusion, int lane, int car_s, size_t prev_size)
{
    carStates car_states = {false, false, false};
    for ( int i = 0; i < sensor_fusion.size(); i++ ) {
        float d = sensor_fusion[i][6];
        int car_lane = -1;
        // is it on the same lane we are
        if ( d > 0 && d < 4 ) {
        car_lane = 0;
        } else if ( d > 4 && d < 8 ) {
        car_lane = 1;
        } else if ( d > 8 && d < 12 ) {
        car_lane = 2;
        }
        if (car_lane < 0) {
        continue;
        }
        // Find car speed.
        double vx = sensor_fusion[i][3];
        double vy = sensor_fusion[i][4];
        double check_speed = sqrt(vx*vx + vy*vy);
        double check_car_s = sensor_fusion[i][5];

        // Estimate car s position after executing previous trajectory.
        check_car_s += ((double)prev_size*0.02*check_speed);

        // A car is considered "dangerous" when its distance to our car is less than 30 meters
        const int MIN_SAFE_DISTANCE = 30;
        
        if ( car_lane == lane ) {
        // Car in our lane.
        car_states.car_ahead |= check_car_s > car_s && check_car_s - car_s < MIN_SAFE_DISTANCE;
        } else if ( car_lane - lane == -1 ) {
        // Car left
        car_states.car_left |= car_s - MIN_SAFE_DISTANCE < check_car_s && car_s + MIN_SAFE_DISTANCE > check_car_s;
        } else if ( car_lane - lane == 1 ) {
        // Car right
        car_states.car_right |= car_s - MIN_SAFE_DISTANCE < check_car_s && car_s + MIN_SAFE_DISTANCE > check_car_s;
        }
    }
    return car_states;
}