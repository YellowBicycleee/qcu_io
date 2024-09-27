#pragma once

#include <cstdint>

namespace base {

// 借鉴KuiperLLama的设计
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};


}