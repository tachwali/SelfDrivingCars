#include "PID.h"

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {    
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;
    sum_cte = 0;
    prev_cte = 0;
}

void PID::UpdateError(double cte) {
    sum_cte += cte;
    p_error = -Kp * cte;
    i_error = -Ki * sum_cte;
    d_error = -Kd * (cte - prev_cte);
    prev_cte = cte;    
}

double PID::TotalError() {
  return p_error + i_error + d_error;
}

