#pragma once
#include <vector>
#include "types.hpp"

using namespace std;

carActions estimate_car_action(int lane, carStates& car_states, const double MAX_ACC);