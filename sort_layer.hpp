#ifndef CAFFE_SORT_LAYER_HPP_
#define CAFFE_SORT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Second-Order Response Transform Layer performs a operation on two input
 *        blobs A and B as @f$ y = A + B + (ReLU(A)*ReLU(B)+shift)^0.5 @f$, where *
 *        represents element-wise production.   
 *          
 *        The backpropagation gradient for bottoms A and B are: 
 *        @f$ y'A = 1 + 0.5(ReLU(A)*ReLU(B)+shift)^(-0.5)*ReLU(B)*(A>0) @f$ 
 *        @f$ y'B = 1 + 0.5(ReLU(A)*ReLU(B)+shift)^(-0.5)*ReLU(A)*(B>0) @f$ 
 */

template <typename Dtype>
class SortLayer : public Layer<Dtype> {
 public:
  explicit SortLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Sort"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  
  Blob<Dtype> bottom_relu;
  Blob<Dtype> bottom_relu_1;
  Blob<Dtype> bottom_gradient;
  Blob<Dtype> bottom_gradient_1;
  Blob<Dtype> after_prod;
  Blob<Dtype> after_sqrt;
  Blob<Dtype> gradient_for_prod;

  Dtype sqrt_shift;
};

}  // namespace caffe

#endif  // CAFFE_SORT_LAYER_HPP_
