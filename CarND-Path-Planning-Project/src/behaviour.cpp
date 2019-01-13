#include "behaviour.hpp"
#include <math.h>
#include "utils.hpp"

carActions estimate_car_action(int lane, carStates& car_states, const double MAX_ACC)
{
    double speed_diff = 0;
    int lane_diff = 0;
    if ( car_states.car_ahead ) { // Car ahead
        if ( !car_states.car_left && lane > 0 ) {
        // if there is no car left and there is a left lane.
        lane_diff = -1; // Change lane left.
        } 
        else if ( !car_states.car_right && lane != 2 ){
        // if there is no car right and there is a right lane.
        lane_diff = +1; // Change lane right.
        } 
        else {
        speed_diff -= MAX_ACC;
        }
    } 
    else {
        if ( lane != 1 ) { // if we are not on the center lane.
            if ( ( lane == 0 && !car_states.car_right ) || ( lane == 2 && !car_states.car_left ) ) {
                lane = 1; // Back to center.
            }
        }

        speed_diff += MAX_ACC;
        // if ( ref_vel < MAX_SPEED ) {
        // speed_diff += MAX_ACC;
        // }
    }

    carActions car_action = {speed_diff, lane_diff};
    return car_action;
}