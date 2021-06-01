//
// Created by Wang, Yao on 5/3/21.
//

#include "op_intensity.h"
#include <tvm/relay/attrs/nn.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/data_layout.h>

namespace tvm {
namespace relay {

int64_t conv2d_ops(const tvm::Array<Type>& types, const Attrs& attrs) {
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  const auto* out = types[2].as<TensorTypeNode>();
  static const tir::Layout kNCHW("NCHW");
  static const tir::Layout kOIHW("OIHW");

  const Conv2DAttrs* param = attrs.as<Conv2DAttrs>();
  ICHECK(param != nullptr);
  const tir::Layout in_layout(param->data_layout);
  const tir::Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCHW);
  if (!trans_in_layout.defined()) {
    LOG(FATAL)
      << "conv2d only support input layouts that are convertible from NCHW."
      << " The provided layout is: " << in_layout;
    return 0;
  }

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIHW);
  if (!trans_kernel_layout.defined()) {
    LOG(FATAL)
      << "conv2d only support kernel layouts that are convertible from OIHW."
      << " The provided layout is: " << kernel_layout;
    return 0;
  }

  tir::Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCHW);
  if (!trans_out_layout.defined()) {
    LOG(FATAL)
      << "conv2d only support output layouts that are convertible from NCHW."
      << "The provided layout is: " << out_layout;
    return 0;
  }

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);
  Array<IndexExpr> wshape_oihw = trans_kernel_layout.ForwardShape(weight->shape);
  Array<IndexExpr> oshape_nchw = trans_out_layout.ForwardShape(out->shape);

  IndexExpr num_ops = 2 * dshape_nchw[0] * dshape_nchw[1] * oshape_nchw[1]
                      * oshape_nchw[2] * oshape_nchw[3] * wshape_oihw[2] * wshape_oihw[3];
  const auto* num_ops_val = num_ops.as<IntImmNode>();

  if (num_ops_val) {
    return num_ops_val->value;
  }

  return 0;
}

int64_t dense_ops(const tvm::Array<Type>& types, const Attrs& attrs) {
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();

  IndexExpr num_ops = 2 * data->shape[0] * data->shape[1] * weight->shape[0];
  const auto* num_ops_val = num_ops.as<IntImmNode>();

  if (num_ops_val) {
    return num_ops_val->value;
  }

  return 0;
}

int64_t batch_matmul_ops(const tvm::Array<Type>& types, const Attrs& attrs) {
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();

  IndexExpr num_ops = 2 * data->shape[0] * data->shape[1] * data->shape[2] * weight->shape[1];
  const auto* num_ops_val = num_ops.as<IntImmNode>();

  if (num_ops_val) {
    return num_ops_val->value;
  }

  return 0;
}

void OpIntensityVisitor::VisitExpr_(const CallNode* node) {
  // Compute data volume.
  size_t num_bits = 0;
  auto types = Array<Type>(node->type_args);
  types.push_back(node->checked_type());
  for (const auto& type : types) {
    const auto* ttype = type.as<TensorTypeNode>();
    if (ttype) {
      size_t shape_mul = 1;
      for (const auto& dim : ttype->shape) {
        const auto* dim_val = dim.as<IntImmNode>();
        if (dim_val) shape_mul *= dim_val->value;
      }
      num_bits += ttype->dtype.bits() * shape_mul;
    }
  }

  mem_vol += num_bits / 8;

  // Compute number of operations.
  const auto* op = node->op.as<OpNode>();
  if (op) {
    if (op->name == "nn.conv2d") num_ops += conv2d_ops(types, node->attrs);
    if (op->name == "nn.dense") num_ops += dense_ops(types, node->attrs);
    if (op->name == "nn.batch_matmul") num_ops += batch_matmul_ops(types, node->attrs);
  }
}

TVM_REGISTER_GLOBAL("relay.analysis.OpIntensity")
    .set_body_typed([](Expr expr) {
      auto op_visitor = OpIntensityVisitor();
      op_visitor.VisitExpr(expr);

      return (float)op_visitor.num_ops / (float)op_visitor.mem_vol;
    });

}  // namespace relay
}  // namespace tvm

