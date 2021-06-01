//
// Created by Wang, Yao on 5/3/21.
//

#ifndef TVM_OP_INTENSITY_H_
#define TVM_OP_INTENSITY_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class OpIntensityVisitor : public MixedModeVisitor {
  public:
    OpIntensityVisitor() : MixedModeVisitor() {}

    void VisitExpr_(const CallNode* node) final;

    // Global variables for memory and operation volume.
    size_t mem_vol = 0;
    size_t num_ops = 0;
};

}
}

#endif //TVM_OP_INTENSITY_H_
