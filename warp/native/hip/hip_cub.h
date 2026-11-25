// SPDX-License-Identifier: Apache-2.0

// ROCm build: CUB is provided by hipCUB, which keeps its interface.

#pragma once

#include "rocprim_radix_sort_fix.h"

#include "../hip_compat.h"
#include <hipcub/hipcub.hpp>

// the form vendored cuBQL also uses, so that both may declare it
namespace cub {
using namespace hipcub;
}
