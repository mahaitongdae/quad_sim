#ifndef __NETWORK_EVALUATE_H__
#define __NETWORK_EVALUATE_H__

#include <math.h>

/*
 * since the network outputs thrust on each motor,
 * we need to define a struct which stores the values
*/
typedef struct control_s {
  union {
    // controlModeLegacy
    struct {
      int16_t roll;
      int16_t pitch;
      int16_t yaw;
      float thrust;
    };

    // controlModeForceTorque
    // Note: Using SI units for a controller makes it hard to tune it for different platforms. The normalized force API
    // is probably a better option.
    struct {
      float thrustSi;  // N
      union { // Nm
        float torque[3];
        struct {
          float torqueX;
          float torqueY;
          float torqueZ;
        };
      };
    };

    // controlModeForce
    float normalizedForces[STABILIZER_NR_OF_MOTORS]; // 0.0 ... 1.0
  };

  control_mode_t controlMode;
} control_t;

void networkEvaluate(control_t *control, const float *state_array);

#endif