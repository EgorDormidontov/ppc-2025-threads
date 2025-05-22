#include "all/dormidontov_e_kannon/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <functional>
#include <vector>

bool dormidontov_e_kannon_all::allTask::PreProcessingImpl() {
  if (world_.rank() == 0) {
    block_size_ = side_size_ / num_blocks_;
    auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
    A_.assign(a_ptr, a_ptr + matrix_size_);
    B_.assign(b_ptr, b_ptr + matrix_size_);
    A_buffer_.assign(matrix_size_, 0);
    B_buffer_.assign(matrix_size_, 0);
    C_.assign(matrix_size_, 0);
  }
  return true;
}

bool dormidontov_e_kannon_all::allTask::ValidationImpl() {
  if (world_.rank() == 0) {
    matrix_size_ = static_cast<size_t>(task_data->inputs_count[0]);
    side_size_ = static_cast<size_t>(std::sqrt(matrix_size_));
    side_size_ = static_cast<size_t>(task_data->inputs_count[2]);
    return task_data->inputs_count[0] == task_data->inputs_count[1] &&
           task_data->outputs_count[0] == task_data->inputs_count[0] && side_size_ % num_blocks_ == 0;
  }
  return true;
}

void dormidontov_e_kannon_all::allTask::StartingShift() {
  std::swap(A_buffer_, A_);
  std::swap(B_buffer_, B_);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_blocks_), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t block_i = r.begin(); block_i != r.end(); ++block_i) {
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

void dormidontov_e_kannon_all::allTask::IterationShift() {
  std::swap(A_buffer_, A_);
  std::swap(B_buffer_, B_);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_blocks_), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t block_i = r.begin(); block_i != r.end(); ++block_i) {
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

void dormidontov_e_kannon_all::allTask::MultImpl() {
  size_t size = world_.size();
  size_t rank = world_.rank();
  std::vector<double> C_local(matrix_size_, 0.0);

  for (size_t iter = 0; iter < num_blocks_; ++iter) {
    size_t blocks_per_proc = (num_blocks_ + size - 1) / size;
    size_t start_block_i = rank * blocks_per_proc;
    size_t end_block_i = std::min(start_block_i + blocks_per_proc, num_blocks_);

    for (size_t block_i = start_block_i; block_i < end_block_i; ++block_i) {
      for (size_t block_j = 0; block_j < num_blocks_; ++block_j) {
        size_t i0 = block_i * block_size_;
        size_t j0 = block_j * block_size_;
        for (size_t i = i0; i < i0 + block_size_; ++i) {
          for (size_t j = j0; j < j0 + block_size_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < block_size_; ++k) {
              sum +=
                  A_[i * side_size_ + (block_j * block_size_ + k)] * B_[(block_i * block_size_ + k) * side_size_ + j];
            }
            C_local[i * side_size_ + j] += sum;
          }
        }
      }
    }

    boost::mpi::all_reduce(world_, C_local.data(), matrix_size_, C_.data(), std::plus<double>{});
  }
}

bool dormidontov_e_kannon_all::allTask::RunImpl() {
  boost::mpi::broadcast(world_, block_size_, 0);
  boost::mpi::broadcast(world_, A_, 0);
  boost::mpi::broadcast(world_, B_, 0);
  boost::mpi::broadcast(world_, A_buffer_, 0);
  boost::mpi::broadcast(world_, B_buffer_, 0);
  boost::mpi::broadcast(world_, C_, 0);

  boost::mpi::broadcast(world_, matrix_size_, 0);
  boost::mpi::broadcast(world_, side_size_, 0);
  boost::mpi::broadcast(world_, side_size_, 0);

  StartingShift();
  for (size_t iter = 0; iter < num_blocks_; ++iter) {
    MultImpl();
    IterationShift();
  }
  return true;
}

bool dormidontov_e_kannon_all::allTask::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}