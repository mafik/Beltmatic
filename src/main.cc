#include <omp.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include "format.hh"
#include "log.hh"
#include "static_vector.hh"
#include "virtual_fs.hh"

#pragma maf main

using namespace stlpb;
using namespace std;
using namespace maf;
using Number = I32;

constexpr Number kExtractors[] = {1,  2,  3,  4,  5,  6,  7,  8,  9, 11,
                                  12, 13, 14, 15, 16, 17, 18, 19, 20};
constexpr int kBeltsPerExtractor = 6;
constexpr int kNExtractors = sizeof(kExtractors) / sizeof(*kExtractors);
constexpr int N = 100001;
constexpr int kMaxCost = 40;

struct Step {
  enum class Type : uint8_t { Extract, Add, Mul, Sub, Sub2, Div, Rem, Exp, Exp2 };
  using enum Type;
  Type type;
  union {
    struct {
      int8_t a, b;  // negative values indicate how far back to move for a given argument
    };
    uint16_t extractor;
  };
};

struct Plan {
  Number value;
  uint8_t cost = 0;
  uint8_t ops = 0;
  uint32_t extractors;
  vector<Step> steps;
  bool operator<(const Plan& other) const { return cost > other.cost; }
};

static constexpr int extractor_cost(int different_extractors) {
  switch (different_extractors) {
    case 0:
    case 1:
      return 0;
    case 2:
      return 3;
    case 3:
      return 6;
    case 4:
      return 9;
    case 5:
      return 12;
    default:
      return kMaxCost;
  }
}

template <typename T>
struct Op {
  static Plan Combine(const Plan& a, const Plan& b) {
    Plan ret = {
        .value = T::Apply(a.value, b.value),
        .cost = 0,
        .ops = U8(a.ops + b.ops + T::extra_ops),
    };
    ret.steps.reserve(a.steps.size() + b.steps.size() + 3);
    ret.steps.insert(ret.steps.end(), a.steps.begin(), a.steps.end());
    ret.steps.insert(ret.steps.end(), b.steps.begin(), b.steps.end());
    T::AddSteps(ret, a, b);
    ret.extractors = a.extractors | b.extractors;
    int different_extractors = popcount(ret.extractors);
    ret.cost = ret.ops + extractor_cost(different_extractors);
    return ret;
  }
};

struct AddOp : Op<AddOp> {
  static const int extra_ops = 1;
  static const Step::Type type = Step::Add;
  static Number Apply(Number a, Number b) {
    auto ret = I64(a) + I64(b);
    if (ret >= N) return 0;
    return ret;
  }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .type = type,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct MulOp : Op<MulOp> {
  static const int extra_ops = 1;
  static const Step::Type type = Step::Mul;
  static Number Apply(Number a, Number b) {
    auto ret = I64(a) * I64(b);
    if (ret >= N) return 0;
    return ret;
  }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
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
        .type = type,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct Sub2Op : Op<Sub2Op> {
  static const int extra_ops = 1;
  static const Step::Type type = Step::Sub2;
  static Number Apply(Number a, Number b) { SubOp::Apply(b, a); }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
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
    if (a == 0) return 0;
    if (a == 1) return 1;
    I64 result = a;
    for (Number i = 1; i < b; ++i) {
      result *= a;
      if (result >= N) return 0;
    }
    return result;
  }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .type = type,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct Exp2Op : Op<Exp2Op> {
  static const int extra_ops = 1;
  static const Step::Type type = Step::Exp2;
  static Number Apply(Number a, Number b) { return ExpOp::Apply(b, a); }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
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
        .type = Step::Div,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
    plan.steps.push_back(Step{
        .type = Step::Rem,
        .a = (int8_t)(-b.steps.size() - 2),
        .b = (int8_t)(-2),
    });
    plan.steps.push_back(Step{
        .type = Base::type,
        .a = (int8_t)(-2),
        .b = (int8_t)(-1),
    });
  }
};

template <typename Base>
struct Div2And : Op<Div2And<Base>> {
  static const int extra_ops = 2;
  static Number Apply(Number a, Number b) { DivAnd<Base>::Apply(b, a); }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .type = Step::Div,
        .a = (int8_t)(-1),
        .b = (int8_t)(-b.steps.size() - 1),
    });
    plan.steps.push_back(Step{
        .type = Step::Rem,
        .a = (int8_t)(-2),
        .b = (int8_t)(-b.steps.size() - 2),
    });
    plan.steps.push_back(Step{
        .type = Base::type,
        .a = (int8_t)(-2),
        .b = (int8_t)(-1),
    });
  }
};

Str ToStr(const Plan& plan, uint8_t step) {
  auto& s = plan.steps[step];

  if (s.type == Step::Extract) {
    return f("%d", s.extractor);
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
  return f("> %d = %s [cost %d]", plan.value, ToStr(plan, plan.steps.size() - 1).c_str(),
           plan.cost);
}
static_vector<Plan, 10> plans[N];

constexpr size_t memory_usage = sizeof(plans) / 1024 / 1024;

vector<Plan> q;
mutex q_mutex;

constexpr int kUniqueSlack = 3;

unordered_set<U64> visited;

static U64 encode(U64 value, U64 extractors, U64 cost) {
  return cost | value << 8 | extractors << 32;
}

static U64 encode(const Plan& plan) { return encode(plan.value, plan.extractors, plan.cost); }

template <typename Op>
void Consider(const Plan& plan_a) {
  auto value_a = plan_a.value;
  vector<Plan> out_plans;

  for (Number value_b = 1; value_b < N; ++value_b) {
    auto new_value = Op::Apply(value_a, value_b);
    if (new_value <= 0 || new_value >= N || new_value == value_a || new_value == value_b) {
      continue;
    }

    if (plans[value_b].empty()) continue;

    auto& other_plans = plans[new_value];
    auto rough_cost_estimate = plan_a.cost + Op::extra_ops - kUniqueSlack;
    if (rough_cost_estimate > kMaxCost) {
      return;
    }
    if (!other_plans.empty() && other_plans.front().cost < rough_cost_estimate) {
      return;
    }
    for (const auto& plan_b : plans[value_b]) {
      auto new_extractors = plan_a.extractors | plan_b.extractors;
      int different_extractors = popcount(new_extractors);
      auto new_cost =
          plan_a.ops + plan_b.ops + Op::extra_ops + extractor_cost(different_extractors);
      if (new_cost > kMaxCost) {
        continue;
      }
      if (visited.count(encode(new_value, new_extractors, new_cost))) {
        continue;
      }

      bool unique = true;
      for (auto& other_plan : other_plans) {
        if (new_extractors == other_plan.extractors) {
          unique = false;
        }
      }
      int slack = unique ? kUniqueSlack : 0;
      if (!other_plans.empty() && other_plans.front().cost < new_cost - slack) {
        continue;
      }
      auto new_plan = Op::Combine(plan_a, plan_b);
      out_plans.push_back(new_plan);
    }
  }

  if (!out_plans.empty()) {
    lock_guard<mutex> lock(q_mutex);
    for (auto& new_plan : out_plans) {
      q.push_back(new_plan);
      push_heap(q.begin(), q.end());
    }
  }
};

int main() {
  q.reserve(N * 10);
  for (int i = 0; i < kNExtractors; ++i) {
    q.push_back(Plan{.value = kExtractors[i],
                     .cost = 1,
                     .ops = 0,
                     .steps = {
                         Step{.type = Step::Extract, .extractor = (U16)kExtractors[i]},
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

    auto key = encode(plan_a);
    if (visited.count(key)) {
      continue;
    }
    visited.insert(key);

    auto value_a = plan_a.value;
    auto& plans_a = plans[value_a];

    bool unique = true;
    for (auto& other_plan : plans_a) {
      if (plan_a.extractors == other_plan.extractors) {
        unique = false;
      }
    }

    int current_best = plans_a.empty() ? kMaxCost + 1 : plans_a.front().cost;
    if (current_best > plan_a.cost) {
      ++improvements;
      plans_a.clear();
      plans_a.push_back(plan_a);
    } else if (current_best == plan_a.cost && unique) {
      plans_a.push_back(plan_a);
    }

    if (unique) {
      // Explore sub-optimal plans, but if they are too bad, skip them
      if (current_best <= plan_a.cost - kUniqueSlack) {
        continue;
      }
    } else {
      if (current_best <= plan_a.cost) {
        continue;
      }
    }

#pragma omp parallel sections
    {
#pragma omp section
      { Consider<AddOp>(plan_a); }
#pragma omp section
      { Consider<MulOp>(plan_a); }
#pragma omp section
      { Consider<SubOp>(plan_a); }
#pragma omp section
      { Consider<Sub2Op>(plan_a); }
#pragma omp section
      { Consider<ExpOp>(plan_a); }
#pragma omp section
      { Consider<Exp2Op>(plan_a); }
#pragma omp section
      { Consider<DivAnd<AddOp>>(plan_a); }
#pragma omp section
      { Consider<DivAnd<MulOp>>(plan_a); }
#pragma omp section
      { Consider<DivAnd<SubOp>>(plan_a); }
#pragma omp section
      { Consider<DivAnd<Sub2Op>>(plan_a); }
#pragma omp section
      { Consider<DivAnd<ExpOp>>(plan_a); }
#pragma omp section
      { Consider<DivAnd<Exp2Op>>(plan_a); }
#pragma omp section
      { Consider<Div2And<AddOp>>(plan_a); }
#pragma omp section
      { Consider<Div2And<MulOp>>(plan_a); }
#pragma omp section
      { Consider<Div2And<SubOp>>(plan_a); }
#pragma omp section
      { Consider<Div2And<Sub2Op>>(plan_a); }
#pragma omp section
      { Consider<Div2And<ExpOp>>(plan_a); }
#pragma omp section
      { Consider<Div2And<Exp2Op>>(plan_a); }
    }
  }

  int solutions_found = 0;
  for (int i = 0; i < N; ++i) {
    if (!plans[i].empty()) {
      ++solutions_found;
    }
  }
  LOG << "Found " << solutions_found << "/" << N - 1 << " solutions";

  Str result;
  for (auto& plan_set : plans) {
    for (auto& plan : plan_set) {
      result += ToStr(plan) + "\n";
      // LOG << plan;
    }
  }

  Status status;
  fs::real.Write(Path("result3.txt"), result, status);
  if (!OK(status)) {
    ERROR << status;
  }
}