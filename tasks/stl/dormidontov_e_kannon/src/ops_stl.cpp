#include "stl/dormidontov_e_kannon/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <core/util/include/util.hpp>
#include <thread>
#include <vector>

bool dormidontov_e_kannon_stl::stlTask::PreProcessingImpl() {
  block_size_ = side_size_ / num_blocks_;

  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  A_.assign(a_ptr, a_ptr + matrix_size_);
  B_.assign(b_ptr, b_ptr + matrix_size_);
  A_buffer_.assign(matrix_size_, 0);
  B_buffer_.assign(matrix_size_, 0);
  C_.assign(matrix_size_, 0);

  return true;
}

bool dormidontov_e_kannon_stl::stlTask::ValidationImpl() {
  matrix_size_ = static_cast<size_t>(task_data->inputs_count[0]);
  side_size_ = static_cast<size_t>(std::sqrt(matrix_size_));
  num_blocks_ = static_cast<size_t>(task_data->inputs_count[2]);

  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->outputs_count[0] == task_data->inputs_count[0] && side_size_ % num_blocks_ == 0;
}

void dormidontov_e_kannon_stl::stlTask::StartingShift() {
  std::swap(A_buffer_, A_);
  std::swap(B_buffer_, B_);

  size_t th_count = std::max(ppc::util::GetPPCNumThreads(), 1);
  th_count = std::min(th_count, num_blocks_);
  std::vector<std::thread> threads(th_count);
  size_t delta = num_blocks_ / th_count;
  size_t rest = num_blocks_ % th_count;
  for (size_t th = 0; th < th_count; ++th) {
    size_t start = delta * th;
    size_t end = delta * (th + 1);
    end += rest;
    if (th != 0) {
      start += rest;
    }
    threads[th] = std::thread([&, start, end]() {
      for (size_t block_i = start; block_i != end; ++block_i) {
        for (size_t block_j = 0; block_j < num_blocks_; ++block_j) {
          size_t row;
          size_t col;
          col = row = (block_j + block_i) % num_blocks_;
          for (size_t i = 0; i < block_size_; ++i) {
            for (size_t j = 0; j < block_size_; ++j) {
              A_[idx(idx(block_i, i, block_size_), idx(block_j, j, block_size_), side_size_)] =
                  A_buffer_[idx(idx(block_i, i, block_size_), idx(col, j, block_size_), side_size_)];
              B_[idx(idx(block_i, i, block_size_), idx(block_j, j, block_size_), side_size_)] =
                  B_buffer_[idx(idx(row, i, block_size_), idx(block_j, j, block_size_), side_size_)];
            }
          }
        }
      }
    });
  }

  for (std::thread& worker : threads) {
    worker.join();
  }
}

void dormidontov_e_kannon_stl::stlTask::IterationShift() {
  std::swap(A_buffer_, A_);
  std::swap(B_buffer_, B_);

  size_t th_count = std::max(ppc::util::GetPPCNumThreads(), 1);
  std::vector<std::thread> threads(th_count);
  size_t delta = num_blocks_ / th_count;
  size_t rest = num_blocks_ % th_count;
  for (size_t th = 0; th < th_count; ++th) {
    size_t start = delta * th;
    size_t end = delta * (th + 1);
    end += rest;
    if (th != 0) {
      start += rest;
    }
    threads[th] = std::thread([&, start, end]() {
      for (size_t block_i = start; block_i != end; ++block_i) {
        for (size_t block_j = 0; block_j < num_blocks_; ++block_j) {
          size_t row;
          size_t col;
          row = (block_i + 1) % num_blocks_;
          col = (block_j + 1) % num_blocks_;
          for (size_t i = 0; i < block_size_; ++i) {
            for (size_t j = 0; j < block_size_; ++j) {
              A_[idx(idx(block_i, i, block_size_), idx(block_j, j, block_size_), side_size_)] =
                  A_buffer_[idx(idx(block_i, i, block_size_), idx(col, j, block_size_), side_size_)];
              B_[idx(idx(block_i, i, block_size_), idx(block_j, j, block_size_), side_size_)] =
                  B_buffer_[idx(idx(row, i, block_size_), idx(block_j, j, block_size_), side_size_)];
            }
          }
        }
      }
    });
  }
  for (std::thread& worker : threads) {
    worker.join();
  }
}

bool dormidontov_e_kannon_stl::stlTask::RunImpl() {
  StartingShift();

  for (size_t iter = 0; iter < num_blocks_; ++iter) {
    for (size_t block_i = 0; block_i < side_size_; block_i += block_size_) {
      for (size_t block_j = 0; block_j < side_size_; block_j += block_size_) {
        for (size_t i = block_i; i < block_i + block_size_; i++) {
          for (size_t j = block_j; j < block_j + block_size_; j++) {
            for (size_t k = 0; k < block_size_; k++) {
              C_[idx(i, j, side_size_)] += A_[idx(i, block_j + k, side_size_)] * B_[idx(block_i + k, j, side_size_)];
            }
          }
        }
      }
    }

    IterationShift();
  }
  return true;
}

bool dormidontov_e_kannon_stl::stlTask::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
