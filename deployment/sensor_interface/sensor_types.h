// sensor_types.h
// Shared sensor reading struct used across I2C / UART / CAN reader modules
// and the main inference loop. Mirrors the observation vector the DQN/LSTM
// were trained on -- keep this in lockstep with env/detector_env.py's
// observation ordering, or the deployed model will silently see garbage.

#pragma once
#include <cstdint>

struct SensorReading {
    double count_rate;        // from detector front-end (counter/detector IC over I2C)
    double gain;               // derived from detector calibration register
    double battery_voltage;    // power monitor (e.g. INA219/INA3221 over I2C)
    double board_temp_c;       // temperature sensor (e.g. onboard thermistor/I2C temp IC)
    double signal_quality;     // derived metric from front-end ADC noise floor
    double vibration;          // motion/IMU magnitude (e.g. accelerometer over I2C/SPI)
    double lstm_fault_prob;    // filled in by the LSTM inference stage, not raw sensor
    double steps_since_recal;  // normalized counter maintained by the control loop
    uint64_t timestamp_ms;
};

// Fixed-point / raw bus read helpers would live in each *_reader.{h,cpp} pair.
