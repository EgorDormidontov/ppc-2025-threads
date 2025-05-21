#pragma once

#include <tbb/tbb.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dormidontov_e_kannon_all {

inline size_t idx(size_t row, size_t column, size_t n) { return (row * n) + column; }

using matrix = std::vector<double>;

matrix GenMatrix(size_t n);
matrix NaiveMultipilication(const matrix& A, const matrix& B, size_t n);

class allTask : public ppc::core::Task {
 public:
  explicit allTask(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  size_t matrix_size_;
  size_t side_size_;
  size_t block_size_;
  size_t num_blocks_;
  matrix A_;
  matrix B_;
  matrix C_;
  matrix A_buffer_;
  matrix B_buffer_;
  boost::mpi::communicator world_;

  void MultImpl();
  void StartingShift();
  void IterationShift();
};
}  // namespace dormidontov_e_kannon_all