#include "hetero_group_completion.hpp"
namespace {
class RunnerGroupBoundary : public HeteroGroupCompletionTest {};
} // namespace
TEST_F(RunnerGroupBoundary, OrdinaryGroupsFenceTheWholeProxySubmission) {
  checkMode(0);
}
TEST_F(RunnerGroupBoundary, DagStreamValueGroupsFenceTheWholeProxySubmission) {
  checkMode(2);
}
