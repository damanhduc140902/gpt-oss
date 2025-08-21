#pragma once

#define NGPU  (2)

struct DeviceTransformer {
  Config config;
  TransformerWeights weights;
  RunState state; // buffers for the "wave" of activations in the forward pass

  float *dev_data; // exclude experts
  float *dev_experts;

  int device_index;

  ~DeviceTransformer();
};