#include <omp.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "format.hh"
#include "log.hh"
#include "static_vector.hh"
#include "virtual_fs.hh"

#pragma maf main

using namespace stlpb;
using namespace std;
using namespace maf;
using Number = I64;

constexpr Number kExtractors[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19};
constexpr int kBeltsPerExtractor = 6;
constexpr int kNExtractors = sizeof(kExtractors) / sizeof(*kExtractors);
constexpr int N = 100001;
constexpr int kMaxCost = 8;

struct Step {
  enum class Type : uint8_t { Extract, Add, Mul, Sub, Sub2, Div, Rem, Exp, Exp2 };
  using enum Type;
  Number value;
  Type type;
  int8_t a, b;  // negative values indicate how far back to move for a given argument
};

struct Plan {
  uint8_t cost = 0;
  uint8_t ops = 0;
  uint32_t extractors;
  static_vector<Step, 20> steps;
  bool operator<(const Plan& other) const { return cost > other.cost; }
  Number Value() const { return steps.back().value; }
};

static bool NumberValid(Number a, Number b, Number result) {
  return result > 0 && result < N && result != a && result != b;
}

template <typename T>
struct Op {
  static Number Valid(Number a, Number b) {
    auto result = T::Apply(a, b);
    if (NumberValid(a, b, result)) {
      return result;
    } else {
      return 0;
    }
  }

  static Plan Combine(const Plan& a, const Plan& b) {
    Plan ret = {
        .cost = 0,
        .ops = U8(a.ops + b.ops + T::extra_ops),
    };
    // ret.steps.reserve(a.steps.size() + b.steps.size() + 3);
    ret.steps.insert(ret.steps.end(), a.steps.begin(), a.steps.end());
    ret.steps.insert(ret.steps.end(), b.steps.begin(), b.steps.end());
    T::AddSteps(ret, a, b);
    ret.extractors = a.extractors | b.extractors;
    int different_extractors = popcount(ret.extractors);
    ret.cost = ret.ops + different_extractors * (different_extractors - 1);
    return ret;
  }
};

struct AddOp : Op<AddOp> {
  static const int extra_ops = 1;
  static const Step::Type type = Step::Add;
  static Number Apply(Number a, Number b) { return a + b; }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .value = Apply(a.Value(), b.Value()),
        .type = type,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct MulOp : Op<MulOp> {
  static const int extra_ops = 1;
  static const Step::Type type = Step::Mul;
  static Number Apply(Number a, Number b) { return a * b; }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .value = Apply(a.Value(), b.Value()),
        .type = type,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct SubOp : Op<SubOp> {
  static const int extra_ops = 1;
  static const Step::Type type = Step::Sub;
  static Number Apply(Number a, Number b) { return a - b; }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .value = Apply(a.Value(), b.Value()),
        .type = type,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct Sub2Op : Op<Sub2Op> {
  static const int extra_ops = 1;
  static const Step::Type type = Step::Sub2;
  static Number Apply(Number a, Number b) { return b - a; }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .value = Apply(a.Value(), b.Value()),
        .type = type,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct ExpOp : Op<ExpOp> {
  static const int extra_ops = 1;
  static const Step::Type type = Step::Exp;
  static Number Apply(Number a, Number b) {
    Number result = 1;
    for (Number i = 0; i < b; ++i) {
      result *= a;
      if (result >= N) return 0;
    }
    return result;
  }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .value = Apply(a.Value(), b.Value()),
        .type = type,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct Exp2Op : Op<Exp2Op> {
  static const int extra_ops = 1;
  static const Step::Type type = Step::Exp2;
  static Number Apply(Number b, Number a) {
    Number result = 1;
    for (Number i = 0; i < b; ++i) {
      result *= a;
      if (result >= N) return 0;
    }
    return result;
  }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .value = Apply(a.Value(), b.Value()),
        .type = type,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

template <typename Base>
struct DivAnd : Op<DivAnd<Base>> {
  static const int extra_ops = 2;
  static Number Apply(Number a, Number b) {
    if (b == 0) return 0;
    return Base::Apply(a / b, a % b);
  }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .value = Number(a.Value() / b.Value()),
        .type = Step::Div,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
    plan.steps.push_back(Step{
        .value = Number(a.Value() % b.Value()),
        .type = Step::Rem,
        .a = (int8_t)(-b.steps.size() - 2),
        .b = (int8_t)(-2),
    });
    plan.steps.push_back(Step{
        .type = Base::type,
        .value = Apply(a.Value(), b.Value()),
        .a = (int8_t)(-2),
        .b = (int8_t)(-1),
    });
  }
};

Str ToStr(const Plan& plan, uint8_t step) {
  auto& s = plan.steps[step];

  if (s.type == Step::Extract) {
    return f("%d", s.value);
  }
  auto a = ToStr(plan, step + s.a);
  auto b = ToStr(plan, step + s.b);
  switch (s.type) {
    case Step::Add: {
      return f("(%s + %s)", a.c_str(), b.c_str());
    }
    case Step::Mul: {
      return f("(%s * %s)", a.c_str(), b.c_str());
    }
    case Step::Sub: {
      return f("(%s - %s)", a.c_str(), b.c_str());
    }
    case Step::Sub2: {
      return f("(%s - %s)", b.c_str(), a.c_str());
    }
    case Step::Div: {
      return f("(%s / %s)", a.c_str(), b.c_str());
    }
    case Step::Rem: {
      return f("(%s %% %s)", a.c_str(), b.c_str());
    }
    case Step::Exp: {
      return f("(%s ^ %s)", a.c_str(), b.c_str());
    }
    case Step::Exp2: {
      return f("(%s ^ %s)", b.c_str(), a.c_str());
    }
    default:
      return "?";
  }
}

Str ToStr(const Plan& plan) {
  return f("> %d = %s [cost %d]", plan.Value(), ToStr(plan, plan.steps.size() - 1).c_str(),
           plan.cost);
}
static_vector<Plan, 10> plans[N];

constexpr size_t memory_usage = sizeof(plans) / 1024 / 1024;

int main() {
  vector<Plan> q;
  q.reserve(N * 10);
  mutex q_mutex;
  for (int i = 0; i < kNExtractors; ++i) {
    q.push_back(Plan{.cost = 1,
                     .ops = 0,
                     .steps = {
                         Step{.type = Step::Extract, .value = kExtractors[i], .a = 0, .b = 0},
                     }});
    q.back().extractors |= 1 << i;
  }

  U64 iteration = 1;
  int improvements = 0;
  constexpr int kLogEvery = 10000;
  auto a = chrono::steady_clock::now();
  while (!q.empty()) {
    pop_heap(q.begin(), q.end());
    auto plan_a = q.back();
    q.pop_back();

    auto value_a = plan_a.Value();
    auto& plans_a = plans[value_a];

    ++iteration;

    if (iteration % kLogEvery == 0) {
      auto b = chrono::steady_clock::now();
      auto elapsed = chrono::duration_cast<chrono::milliseconds>(b - a).count();
      a = b;
      double rate = kLogEvery / (elapsed / 1000.0);
      LOG << "Iteration " << iteration << ". Queue size = " << q.size() << ". Rate = " << rate
          << " it/s. Improvements = " << improvements << ". Current cost = " << plan_a.cost;
      improvements = 0;
    }

    if (plans_a.empty()) {
      ++improvements;
    } else {
      if (plans_a.front().cost > plan_a.cost) {
        plans_a.clear();
        ++improvements;
      } else if (plans_a.front().cost < plan_a.cost) {
        continue;
      } else if (plans_a.front().cost == plan_a.cost) {
        bool redundant = false;
        for (auto& other_plan : plans_a) {
          if (plan_a.extractors == other_plan.extractors) {
            redundant = true;
          }
        }
        if (redundant) {
          continue;
        }
      }
    }
    plans_a.push_back(plan_a);

#pragma omp parallel for
    for (Number value_b = 1; value_b < N; ++value_b) {
      auto& plans_b = plans[value_b];
      if (plans_b.empty()) continue;

      auto Consider = [&](auto op) {
        auto new_value = op.Valid(value_a, value_b);
        if (new_value == 0) {
          return;
        }
        auto& other_plans = plans[new_value];
        auto rough_cost_estimate = plan_a.cost + op.extra_ops + 1;
        if (rough_cost_estimate > kMaxCost) {
          return;
        }
        if (!other_plans.empty() && other_plans.front().cost < rough_cost_estimate) {
          return;
        }
        for (auto& plan_b : plans_b) {
          // optimistic estimate, assuming extractors combine perfectly
          // without increasing cost
          auto new_cost = plan_a.ops + plan_b.ops + op.extra_ops;
          int different_extractors = popcount(plan_a.extractors | plan_b.extractors);
          new_cost += different_extractors * (different_extractors - 1);
          if (new_cost > kMaxCost) {
            continue;
          }
          if (!other_plans.empty() && other_plans.front().cost < new_cost) {
            continue;
          }
          auto new_plan = op.Combine(plan_a, plan_b);
          {
            lock_guard<mutex> lock(q_mutex);
            q.push_back(new_plan);
            push_heap(q.begin(), q.end());
          }
        }
      };

      Consider(AddOp{});
      Consider(MulOp{});
      Consider(SubOp{});
      Consider(Sub2Op{});
      Consider(ExpOp{});
      Consider(Exp2Op{});
      Consider(DivAnd<AddOp>{});
      Consider(DivAnd<MulOp>{});
      Consider(DivAnd<SubOp>{});
      Consider(DivAnd<Sub2Op>{});
      Consider(DivAnd<ExpOp>{});
      Consider(DivAnd<Exp2Op>{});
    }
  }

  Str result;
  for (auto& plan_set : plans) {
    for (auto& plan : plan_set) {
      result += ToStr(plan) + "\n";
      LOG << plan;
    }
  }

  Status status;
  fs::real.Write(Path("result3.txt"), result, status);
  if (!OK(status)) {
    ERROR << status;
  }
}