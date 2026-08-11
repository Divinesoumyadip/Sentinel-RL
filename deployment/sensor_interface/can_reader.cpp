// can_reader.cpp
//
// CAN bus interface using Linux SocketCAN (standard on Jetson carrier boards
// with a CAN transceiver, e.g. MCP2515 over SPI or onboard CAN controller).
// Used here for the vibration/motion sensor node and for sending the RL
// agent's chosen action out to the actuator node (power-mode relay,
// recalibration trigger, maintenance-flag indicator) -- CAN's multi-master,
// error-frame-based arbitration is why it's the standard choice for
// fault-tolerant multi-node embedded systems, which is exactly the reason
// it's used for the actuator command path here rather than a simple GPIO.

#include "sensor_types.h"
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <linux/can.h>
#include <linux/can/raw.h>

namespace can_bus {

constexpr canid_t CAN_ID_VIBRATION_SENSOR = 0x100;
constexpr canid_t CAN_ID_ACTUATOR_CMD      = 0x200;

class CANSocket {
public:
    explicit CANSocket(const char* iface = "can0") {
        sock_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (sock_ < 0) {
            std::fprintf(stderr, "WARN: could not create CAN socket (%s) -- "
                         "simulated mode.\n", std::strerror(errno));
            return;
        }
        ifreq ifr{};
        std::strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
        if (ioctl(sock_, SIOCGIFINDEX, &ifr) < 0) {
            std::fprintf(stderr, "WARN: CAN interface %s not found -- simulated mode.\n", iface);
            close(sock_);
            sock_ = -1;
            return;
        }
        sockaddr_can addr{};
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
        if (bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::fprintf(stderr, "WARN: CAN bind failed -- simulated mode.\n");
            close(sock_);
            sock_ = -1;
        }
    }

    ~CANSocket() { if (sock_ >= 0) close(sock_); }
    bool ok() const { return sock_ >= 0; }

    bool read_frame(can_frame* frame) {
        if (sock_ < 0) return false;
        ssize_t nbytes = read(sock_, frame, sizeof(can_frame));
        return nbytes == sizeof(can_frame);
    }

    bool send_frame(const can_frame& frame) {
        if (sock_ < 0) return false;
        return write(sock_, &frame, sizeof(can_frame)) == sizeof(can_frame);
    }

private:
    int sock_ = -1;
};

// Reads vibration magnitude from the CAN-attached motion sensor node.
// Payload convention: bytes[0..3] = little-endian float32 (mg units).
bool read_vibration(CANSocket& bus, double* vibration_out) {
    can_frame frame{};
    if (!bus.read_frame(&frame)) return false;
    if (frame.can_id != CAN_ID_VIBRATION_SENSOR || frame.can_dlc < 4) return false;
    float raw;
    std::memcpy(&raw, frame.data, sizeof(float));
    *vibration_out = static_cast<double>(raw) / 1000.0;  // mg -> g
    return true;
}

// Publishes the RL agent's chosen action to the actuator node over CAN.
bool send_action_command(CANSocket& bus, int action_id) {
    can_frame frame{};
    frame.can_id = CAN_ID_ACTUATOR_CMD;
    frame.can_dlc = 1;
    frame.data[0] = static_cast<uint8_t>(action_id);
    return bus.send_frame(frame);
}

}  // namespace can_bus
