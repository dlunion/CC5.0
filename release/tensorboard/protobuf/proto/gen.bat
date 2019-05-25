@echo off

protoc tensorboardX/proto/attr_value.proto --cpp_out=./
protoc tensorboardX/proto/event.proto --cpp_out=./
protoc tensorboardX/proto/graph.proto --cpp_out=./
protoc tensorboardX/proto/layout.proto --cpp_out=./
protoc tensorboardX/proto/node_def.proto --cpp_out=./
protoc tensorboardX/proto/plugin_pr_curve.proto --cpp_out=./
protoc tensorboardX/proto/plugin_text.proto --cpp_out=./
protoc tensorboardX/proto/resource_handle.proto --cpp_out=./
protoc tensorboardX/proto/step_stats.proto --cpp_out=./
protoc tensorboardX/proto/summary.proto --cpp_out=./
protoc tensorboardX/proto/tensor.proto --cpp_out=./
protoc tensorboardX/proto/tensor_shape.proto --cpp_out=./
protoc tensorboardX/proto/types.proto --cpp_out=./
protoc tensorboardX/proto/versions.proto --cpp_out=./