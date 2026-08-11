// i2c_reader.cpp
//
// I2C sensor interface for the Jetson's onboard/attached sensors:
//   - power monitor (e.g. INA219-style: voltage/current registers)
//   - temperature sensor
//   - detector front-end gain/count registers
//
// Uses Linux's i2c-dev interface (/dev/i2c-N), the standard approach on
// Jetson-class boards. This is a functional skeleton: register maps below
// are illustrative placeholders (INA219-style addressing) -- swap in the
// real datasheet register map for the production part.

#include "sensor_types.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <chrono>

namespace i2c {

constexpr uint8_t ADDR_POWER_MONITOR = 0x40;   // typical INA219 default addr
constexpr uint8_t ADDR_TEMP_SENSOR   = 0x48;   // typical TMP1xx/LM75 default addr
constexpr uint8_t ADDR_DETECTOR_FE   = 0x50;   // placeholder detector front-end addr

constexpr uint8_t REG_BUS_VOLTAGE    = 0x02;
constexpr uint8_t REG_TEMP           = 0x00;
constexpr uint8_t REG_COUNT_RATE     = 0x10;
constexpr uint8_t REG_GAIN           = 0x11;

class I2CBus {
public:
    explicit I2CBus(const char* device_path = "/dev/i2c-1") {
        fd_ = open(device_path, O_RDWR);
        if (fd_ < 0) {
            std::fprintf(stderr, "WARN: could not open %s (%s) -- running in "
                         "simulated-readback mode.\n", device_path, std::strerror(errno));
        }
    }

    ~I2CBus() {
        if (fd_ >= 0) close(fd_);
    }

    bool ok() const { return fd_ >= 0; }

    // Reads a 16-bit big-endian register from a device at `addr`.
    // Returns false (and leaves *out untouched) on bus error.
    bool read_reg16(uint8_t addr, uint8_t reg, uint16_t* out) {
        if (fd_ < 0) return false;
        if (ioctl(fd_, I2C_SLAVE, addr) < 0) return false;
        if (write(fd_, &reg, 1) != 1) return false;
        uint8_t buf[2];
        if (read(fd_, buf, 2) != 2) return false;
        *out = (static_cast<uint16_t>(buf[0]) << 8) | buf[1];
        return true;
    }

private:
    int fd_ = -1;
};

// Reads all I2C-attached sensors into a partially-populated SensorReading.
// UART/CAN readers (see uart_reader.cpp / can_reader.cpp) fill the rest.
// On bus failure, falls back to a last-known-good / safe-default value and
// logs a warning rather than crashing the control loop -- fault tolerance
// starts at the sensor-read layer, not just in the RL policy.
bool read_i2c_sensors(I2CBus& bus, SensorReading* reading) {
    bool all_ok = true;
    uint16_t raw;

    if (bus.read_reg16(ADDR_POWER_MONITOR, REG_BUS_VOLTAGE, &raw)) {
        reading->battery_voltage = (raw >> 3) * 0.004;  // INA219-style LSB scaling
    } else {
        reading->battery_voltage = 7.4;  // safe fallback midpoint, flagged via all_ok
        all_ok = false;
    }

    if (bus.read_reg16(ADDR_TEMP_SENSOR, REG_TEMP, &raw)) {
        reading->board_temp_c = static_cast<int16_t>(raw) / 256.0;
    } else {
        reading->board_temp_c = 25.0;
        all_ok = false;
    }

    if (bus.read_reg16(ADDR_DETECTOR_FE, REG_COUNT_RATE, &raw)) {
        reading->count_rate = static_cast<double>(raw);
    } else {
        reading->count_rate = 0.0;
        all_ok = false;
    }

    if (bus.read_reg16(ADDR_DETECTOR_FE, REG_GAIN, &raw)) {
        reading->gain = raw / 1000.0;
    } else {
        reading->gain = 1.0;
        all_ok = false;
    }

    reading->timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    return all_ok;
}

}  // namespace i2c
