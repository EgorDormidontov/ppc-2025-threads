#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/dormidontov_e_kannon/include/ops_stl.hpp"

namespace dormidontov_e_kannon_stl {
matrix GenMatrix(size_t n) {
  matrix mat(n * n);
  std::mt19937 gen(1337);
  std::uniform_real_distribution<double> distribution(-100, 100);

  for (size_t i = 0; i < n * n; i++) {
    mat[i] = distribution(gen);
  }
  return mat;
}

matrix NaiveMultipilication(const matrix& A, const matrix& B, size_t n) {
  matrix C(n * n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < n; k++) {
        C[idx(i, j, n)] += A[idx(i, k, n)] * B[idx(k, j, n)];
      }
    }
  }
  return C;
}

TEST(dormidontov_e_kannon_stl, mat16) {
  size_t test_side_size = 4;
  size_t test_num_blocks = 2;
  matrix A = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1};
  matrix B = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  matrix ans = NaiveMultipilication(A, B, test_side_size);
  matrix C(test_side_size * test_side_size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->inputs_count.emplace_back(test_num_blocks);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  dormidontov_e_kannon_stl::stlTask task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (size_t i = 0; i < test_side_size * test_side_size; i++) {
    EXPECT_EQ(ans[i], C[i]);
  }
}

TEST(dormidontov_e_kannon_stl, wrong_matrix_size) {
  matrix A(4 * 4);
  matrix B(3 * 3);
  matrix C(2 * 2);
  size_t test_num_blocks = 1;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->inputs_count.emplace_back(test_num_blocks);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  dormidontov_e_kannon_stl::stlTask task_seq(task_data_seq);
  ASSERT_FALSE(task_seq.Validation());
}

TEST(dormidontov_e_kannon_stl, wrong_block_size) {
  size_t test_side_size = 17;
  size_t test_num_blocks = 8;
  matrix A(test_side_size * test_side_size, 1.0);
  matrix B(test_side_size * test_side_size, 1.0);
  matrix C(test_side_size * test_side_size);
  matrix ans(test_side_size * test_side_size, test_side_size);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->inputs_count.emplace_back(test_num_blocks);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  dormidontov_e_kannon_stl::stlTask task_seq(task_data_seq);
  ASSERT_FALSE(task_seq.Validation());
}

TEST(dormidontov_e_kannon_stl, mat27x27) {
  size_t test_side_size = 27;
  size_t test_num_blocks = 3;
  matrix A(test_side_size * test_side_size, 1);
  matrix B(test_side_size * test_side_size, 1);
  matrix C(test_side_size * test_side_size, 0);
  matrix ans = NaiveMultipilication(A, B, test_side_size);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->inputs_count.emplace_back(test_num_blocks);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  dormidontov_e_kannon_stl::stlTask task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (size_t i = 0; i < test_side_size * test_side_size; i++) {
    EXPECT_NEAR(ans[i], C[i], 1e-6);
  }
}
}  // namespace dormidontov_e_kannon_stl
