#pragma once

#include <vector>
#include "types.hpp"

carStates predict(std::vector<std::vector<float>> sensor_fusion, int lane, int car_s, size_t prev_size);
