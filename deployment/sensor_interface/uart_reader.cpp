// uart_reader.cpp
//
// UART interface to the detector's signal-conditioning front-end, which
// streams a signal-quality metric (SNR-derived) as a newline-terminated
// ASCII line, e.g. "SQ:0.932\n". Many detector front-ends expose a debug/
// telemetry UART separate from the I2C register bus -- this module reads
// that stream and does basic noise filtering (rolling median) before it
// reaches the model, mirroring data/data_pipeline.py's denoise() step but
// implemented for the real-time C++ path.

#include "sensor_types.h"
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <deque>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace uart {

class UARTPort {
public:
    explicit UARTPort(const char* device_path = "/dev/ttyTHS1", int baud = B115200) {
        fd_ = open(device_path, O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd_ < 0) {
            std::fprintf(stderr, "WARN: could not open %s (%s) -- simulated mode.\n",
                         device_path, std::strerror(errno));
            return;
        }
        termios tty{};
        tcgetattr(fd_, &tty);
        cfsetospeed(&tty, baud);
        cfsetispeed(&tty, baud);
        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
        tty.c_iflag &= ~IGNBRK;
        tty.c_lflag = 0;
        tty.c_oflag = 0;
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 1;  // 100ms read timeout
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~(PARENB | PARODD);
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;
        tcsetattr(fd_, TCSANOW, &tty);
    }

    ~UARTPort() { if (fd_ >= 0) close(fd_); }
    bool ok() const { return fd_ >= 0; }

    // Reads one line (up to '\n'), returns false on timeout/error.
    bool read_line(char* buf, size_t buf_size) {
        if (fd_ < 0) return false;
        size_t n = 0;
        while (n < buf_size - 1) {
            char c;
            int r = read(fd_, &c, 1);
            if (r <= 0) return false;
            if (c == '\n') break;
            buf[n++] = c;
        }
        buf[n] = '\0';
        return n > 0;
    }

private:
    int fd_ = -1;
};

// Rolling-median noise filter matching the training-time denoise() logic,
// so live inference sees data statistically consistent with training data.
class RollingMedianFilter {
public:
    explicit RollingMedianFilter(size_t window = 5) : window_(window) {}

    double push(double value) {
        buf_.push_back(value);
        if (buf_.size() > window_) buf_.pop_front();
        std::vector<double> sorted(buf_.begin(), buf_.end());
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() / 2];
    }

private:
    size_t window_;
    std::deque<double> buf_;
};

// Parses "SQ:0.932" style lines into a filtered signal_quality value.
bool read_uart_signal_quality(UARTPort& port, RollingMedianFilter& filt, double* out) {
    char line[64];
    if (!port.read_line(line, sizeof(line))) return false;
    double raw;
    if (std::sscanf(line, "SQ:%lf", &raw) != 1) return false;
    *out = filt.push(raw);
    return true;
}

}  // namespace uart
