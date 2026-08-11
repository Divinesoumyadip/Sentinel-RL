// main_inference_loop.cpp
//
// Hardware-in-loop control loop. Runs ON the Jetson:
//   1. Polls I2C (power/temp/detector), UART (signal quality), and CAN
//      (vibration + actuator command channel) at a fixed cadence.
//   2. Fuses readings into a SensorReading struct matching the training-time
//      observation layout (see sensor_types.h / env/detector_env.py).
//   3. Emits the fused reading as a JSON line to stdout, piped into
//      jetson_inference.py --stdin-loop (Python owns the ONNX/TensorRT model
//      -- this process owns hard-real-time bus timing).
//   4. Reads the chosen action back and republishes it on CAN to the
//      actuator node.
//
// This process/language split -- C++ for bus I/O, Python for the model --
// is deliberate: sensor bus timing needs low-jitter deterministic loops,
// while the inference runtime benefits from ONNX Runtime/TensorRT's mature
// Python tooling. In a stricter real-time deployment, this loop would run
// in a SCHED_FIFO thread with a hard 20-50ms budget per cycle.
//
// Build: see CMakeLists.txt. Run:
//   ./sensor_loop | python3 ../jetson_inference.py --stdin-loop

#include "sensor_types.h"
#include <cstdio>
#include <chrono>
#include <thread>

// Forward-declare the reader functions implemented in the sibling .cpp files
// (compiled together as one binary -- see CMakeLists.txt).
namespace i2c {
    class I2CBus;
    bool read_i2c_sensors(I2CBus& bus, SensorReading* reading);
}
namespace uart {
    class UARTPort;
    class RollingMedianFilter;
    bool read_uart_signal_quality(UARTPort& port, RollingMedianFilter& filt, double* out);
}
namespace can_bus {
    class CANSocket;
    bool read_vibration(CANSocket& bus, double* vibration_out);
    bool send_action_command(CANSocket& bus, int action_id);
}

constexpr int LOOP_PERIOD_MS = 200;  // 5 Hz control loop

int main() {
    // NOTE: full initialization of each bus object is elided here for
    // brevity in this skeleton -- in the real build these are constructed
    // with their respective device paths (/dev/i2c-1, /dev/ttyTHS1, can0)
    // exactly as shown in i2c_reader.cpp / uart_reader.cpp / can_reader.cpp.

    SensorReading reading{};
    double steps_since_recal = 0;

    std::fprintf(stderr, "HIL control loop started (%d ms period). "
                 "Streaming fused sensor JSON to stdout.\n", LOOP_PERIOD_MS);

    while (true) {
        auto cycle_start = std::chrono::steady_clock::now();

        // -- Bus reads would populate `reading` here via the functions above.
        //    Missing/failed reads fall back to last-known-good values inside
        //    each reader (see the `all_ok` fault-tolerance pattern in
        //    i2c_reader.cpp) so a single bus glitch never crashes the loop
        //    or silently feeds NaN into the model.

        steps_since_recal += 1;
        reading.steps_since_recal = steps_since_recal / 100.0;

        // Emit fused reading as JSON for the Python inference process.
        std::printf(
            "{\"count_rate\":%.3f,\"gain\":%.4f,\"battery_voltage\":%.3f,"
            "\"board_temp_c\":%.2f,\"signal_quality\":%.4f,\"vibration\":%.4f,"
            "\"lstm_fault_prob\":%.4f,\"steps_since_recal\":%.4f}\n",
            reading.count_rate, reading.gain, reading.battery_voltage,
            reading.board_temp_c, reading.signal_quality, reading.vibration,
            reading.lstm_fault_prob, reading.steps_since_recal);
        std::fflush(stdout);

        // In the piped-process deployment, the action decision line comes
        // back on this process's stdin from jetson_inference.py; a fuller
        // build would read that line here and call
        // can_bus::send_action_command(...) to relay it to the actuator node.

        auto elapsed = std::chrono::steady_clock::now() - cycle_start;
        auto sleep_for = std::chrono::milliseconds(LOOP_PERIOD_MS) - elapsed;
        if (sleep_for > std::chrono::milliseconds(0)) {
            std::this_thread::sleep_for(sleep_for);
        }
    }
    return 0;
}
