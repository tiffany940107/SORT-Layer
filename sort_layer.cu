#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sort_layer.hpp"
#include "caffe/util/math_functions.hpp"



namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out
    ) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : 0;
  }
}

template <typename Dtype>
void SortLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_data_1 = bottom[1]->gpu_data();
  Dtype* bottom_relu_data = bottom_relu.mutable_gpu_data();
  Dtype* bottom_relu_data_1 = bottom_relu_1.mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype a=0.5;
  Dtype b=1.0;
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom_relu_data);
  
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data_1, bottom_relu_data_1);
  
  //for (int i = 0; i < count; ++i) {
  //  bottom_relu_data[i]=std::max(bottom_data[i],Dtype(0));
  //  bottom_relu_data_1[i]=std::max(bottom_data_1[i],Dtype(0));
  //}
  caffe_gpu_mul(count, bottom_relu.gpu_data(), bottom_relu_1.gpu_data(), after_prod.mutable_gpu_data());
  caffe_gpu_scale(count, b, after_prod.gpu_data(), after_sqrt.mutable_gpu_data());
  caffe_gpu_add_scalar(count, sqrt_shift, after_sqrt.mutable_gpu_data());
  caffe_gpu_powx(count, after_sqrt.mutable_gpu_data(), a, after_sqrt.mutable_gpu_data());
  caffe_gpu_add(count, bottom_data, bottom_data_1, top_data);
  caffe_gpu_add(count, top_data, after_sqrt.gpu_data(), top_data);
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, const Dtype* in_data_1, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (1.0 + in_data_1[index]*(in_data[index] > 0));
  }
}

template <typename Dtype>
void SortLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_data_1 = bottom[1]->gpu_data();

    Dtype* bottom_relu_data = bottom_relu.mutable_gpu_data();
    Dtype* bottom_relu_data_1 = bottom_relu_1.mutable_gpu_data();


    const Dtype* top_diff = top[0]->gpu_diff();

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff_1 = bottom[1]->mutable_gpu_diff();
    
    Dtype* bottom_gradient_data = bottom_gradient.mutable_gpu_data();
    Dtype* bottom_gradient_data_1 = bottom_gradient_1.mutable_gpu_data();
     
    Dtype a=0.5;
    Dtype b=-0.5;
    Dtype c=1.0;

    const int count = bottom[0]->count();
    ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom_relu_data);
    ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data_1, bottom_relu_data_1);
    //for (int i = 0; i < count; ++i) {
    //  bottom_relu_data[i]=std::max(bottom_data[i],Dtype(0));
    //  bottom_relu_data_1[i]=std::max(bottom_data_1[i],Dtype(0));
    //}
    caffe_gpu_mul(count, bottom_relu.gpu_data(), bottom_relu_1.gpu_data(), after_prod.mutable_gpu_data());

    caffe_gpu_scale(count, c, after_prod.gpu_data(), gradient_for_prod.mutable_gpu_data());
    caffe_gpu_add_scalar(count, sqrt_shift, gradient_for_prod.mutable_gpu_data());
    caffe_gpu_powx(count, gradient_for_prod.mutable_gpu_data(), b, gradient_for_prod.mutable_gpu_data());
    caffe_gpu_scal(count, a, gradient_for_prod.mutable_gpu_data());
    
    caffe_gpu_mul(count, gradient_for_prod.gpu_data(), bottom_relu_1.gpu_data(),bottom_gradient.mutable_gpu_data());
    caffe_gpu_mul(count, gradient_for_prod.gpu_data(), bottom_relu.gpu_data(),bottom_gradient_1.mutable_gpu_data());
    
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_gradient_data, bottom_diff);
    
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data_1, bottom_gradient_data_1, bottom_diff_1);
   

    //for (int i = 0; i < count; ++i) {
    //  bottom_diff[i] = top_diff[i] * (1.0+bottom_gradient_data[i]*(bottom_data[i] > 0));
    //  bottom_diff_1[i] = top_diff[i] * (1.0+bottom_gradient_data_1[i]*(bottom_data_1[i] > 0));    
    //}
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SortLayer);

}  // namespace caffe
