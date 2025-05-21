#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/dormidontov_e_kannon/include/ops_stl.hpp"

TEST(dormidontov_e_kannon_stl, test_pipeline_run) {
  size_t test_side_size = 300;
  size_t test_num_blocks = 30;
  std::vector<double> a(test_side_size * test_side_size, 1.0);
  std::vector<double> b(test_side_size * test_side_size, 1.0);
  std::vector<double> c(test_side_size * test_side_size);
  std::vector<double> ans(test_side_size * test_side_size, test_side_size);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->inputs_count.emplace_back(test_num_blocks);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  auto task_seq = std::make_shared<dormidontov_e_kannon_stl::stlTask>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_seq);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (size_t i = 0; i < test_side_size * test_side_size; i++) {
    ASSERT_EQ(ans[i], c[i]);
  }
}

TEST(dormidontov_e_kannon_stl, test_task_run) {
  size_t test_side_size = 300;
  size_t test_num_blocks = 30;
  std::vector<double> a(test_side_size * test_side_size, 1.0);
  std::vector<double> b(test_side_size * test_side_size, 1.0);
  std::vector<double> c(test_side_size * test_side_size, 0.0);
  std::vector<double> ans(test_side_size * test_side_size, test_side_size);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->inputs_count.emplace_back(test_num_blocks);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  auto task_seq = std::make_shared<dormidontov_e_kannon_stl::stlTask>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_seq);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (size_t i = 0; i < test_side_size * test_side_size; i++) {
    ASSERT_EQ(ans[i], c[i]);
  }
}