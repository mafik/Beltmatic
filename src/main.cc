#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "format.hh"
#include "log.hh"
#include "virtual_fs.hh"

#pragma maf main

using namespace std;
using namespace maf;
using Number = I16;

constexpr Number kExtractors[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12};
constexpr int kBeltsPerExtractor = 4;
constexpr int kNExtractors = sizeof(kExtractors) / sizeof(*kExtractors);

struct Step {
  enum class Type : uint8_t { Extract, Add, Mul, Sub, Div, Rem };
  using enum Type;
  Type type;
  Number value;
  int8_t a, b;  // negative values indicate how far back to move for a given argument
};

struct Plan {
  uint8_t cost = 0;
  uint8_t ops = 0;
  uint8_t extractor_count[kNExtractors] = {};
  vector<Step> steps;
  bool operator<(const Plan& other) const { return cost < other.cost; }
  bool SameExtractors(const Plan& other) const {
    return memcmp(extractor_count, other.extractor_count, sizeof(extractor_count)) == 0;
  }
  Number Value() const { return steps.back().value; }
};

constexpr int N = 10001;
constexpr int kMaxCost = 9;

enum class Validity {
  Invalid,
  OK,
  Reverse,
};

static bool NumberValid(Number a, Number b, Number result) {
  return result > 0 && result < N && result != a && result != b;
}

template <typename T>
struct Op {
  static Validity Valid(Number a, Number b) {
    if (NumberValid(a, b, T::Apply(a, b))) {
      return Validity::OK;
    } else if (NumberValid(b, a, T::Apply(b, a))) {
      return Validity::Reverse;
    } else {
      return Validity::Invalid;
    }
  }

  static Plan Combine(const Plan& a, const Plan& b) {
    Plan ret = {
        .cost = 0,
        .ops = U8(a.ops + b.ops + T::extra_ops),
    };
    ret.steps.reserve(a.steps.size() + b.steps.size() + 3);
    ret.steps.insert(ret.steps.end(), a.steps.begin(), a.steps.end());
    ret.steps.insert(ret.steps.end(), b.steps.begin(), b.steps.end());
    T::AddSteps(ret, a, b);
    ret.cost += ret.ops;
    for (int i = 0; i < kNExtractors; ++i) {
      auto x = a.extractor_count[i] + b.extractor_count[i];
      ret.extractor_count[i] = x;
      ret.cost += (x + kBeltsPerExtractor - 1) / kBeltsPerExtractor;  // rounding up
    }
    return ret;
  }
};

struct AddOp : Op<AddOp> {
  static const int extra_ops = 1;
  static Number Apply(Number a, Number b) { return a + b; }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .type = Step::Add,
        .value = Apply(a.Value(), b.Value()),
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct MulOp : Op<MulOp> {
  static const int extra_ops = 1;
  static Number Apply(Number a, Number b) { return a * b; }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .type = Step::Mul,
        .value = Apply(a.Value(), b.Value()),
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct SubOp : Op<SubOp> {
  static const int extra_ops = 1;
  static Number Apply(Number a, Number b) { return a - b; }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .type = Step::Sub,
        .value = Apply(a.Value(), b.Value()),
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
  }
};

struct DivPlusRemOp : Op<DivPlusRemOp> {
  static const int extra_ops = 2;
  static Number Apply(Number a, Number b) {
    if (b == 0) return 0;
    return a / b + a % b;
  }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .type = Step::Div,
        .value = Number(a.Value() / b.Value()),
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
    plan.steps.push_back(Step{
        .type = Step::Rem,
        .value = Number(a.Value() % b.Value()),
        .a = (int8_t)(-b.steps.size() - 2),
        .b = (int8_t)(-2),
    });
    plan.steps.push_back(Step{
        .type = Step::Add,
        .value = Apply(a.Value(), b.Value()),
        .a = (int8_t)(-2),
        .b = (int8_t)(-1),
    });
  }
};

struct DivMulRemOp : Op<DivMulRemOp> {
  static const int extra_ops = 2;
  static Number Apply(Number a, Number b) {
    if (b == 0) return 0;
    return (a / b) * (a % b);
  }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    plan.steps.push_back(Step{
        .type = Step::Div,
        .value = Number(a.Value() / b.Value()),
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
    plan.steps.push_back(Step{
        .type = Step::Rem,
        .value = Number(a.Value() % b.Value()),
        .a = (int8_t)(-b.steps.size() - 2),
        .b = (int8_t)(-2),
    });
    plan.steps.push_back(Step{
        .type = Step::Mul,
        .value = Apply(a.Value(), b.Value()),
        .a = (int8_t)(-2),
        .b = (int8_t)(-1),
    });
  }
};

struct DivSubRemOp : Op<DivSubRemOp> {
  static const int extra_ops = 2;
  static Number Apply(Number a, Number b) {
    if (b == 0) return 0;
    auto div = a / b;
    auto rem = a % b;
    if (div > rem) {
      return div - rem;
    } else {
      return rem - div;
    }
  }

  static void AddSteps(Plan& plan, const Plan& a, const Plan& b) {
    Number div = a.Value() / b.Value();
    Number rem = a.Value() % b.Value();
    plan.steps.push_back(Step{
        .type = Step::Div,
        .value = div,
        .a = (int8_t)(-b.steps.size() - 1),
        .b = (int8_t)(-1),
    });
    plan.steps.push_back(Step{
        .type = Step::Rem,
        .value = rem,
        .a = (int8_t)(-b.steps.size() - 2),
        .b = (int8_t)(-2),
    });
    if (div > rem) {
      plan.steps.push_back(Step{
          .type = Step::Mul,
          .value = Number(div - rem),
          .a = (int8_t)(-2),
          .b = (int8_t)(-1),
      });
    } else {
      plan.steps.push_back(Step{
          .type = Step::Mul,
          .value = Number(rem - div),
          .a = (int8_t)(-1),
          .b = (int8_t)(-2),
      });
    }
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
    case Step::Div: {
      return f("(%s / %s)", a.c_str(), b.c_str());
    }
    case Step::Rem: {
      return f("(%s %% %s)", a.c_str(), b.c_str());
    }
    default:
      return "?";
  }
}

Str ToStr(const Plan& plan) {
  return f("%d = %s [cost %d]", plan.Value(), ToStr(plan, plan.steps.size() - 1).c_str(),
           plan.cost);
}
vector<Plan> plans[N];

int main() {
  deque<Plan> q;
  for (int i = 0; i < kNExtractors; ++i) {
    q.push_back(Plan{.cost = 0,
                     .ops = 0,
                     .steps = {
                         Step{.type = Step::Extract, .value = kExtractors[i], .a = 0, .b = 0},
                     }});
    q.back().extractor_count[i]++;
  }

  while (!q.empty()) {
    auto plan_a = q.front();
    auto value_a = plan_a.steps.back().value;
    auto& plans_a = plans[value_a];
    q.pop_front();

    if (q.size() % 10000 == 0) LOG << q.size();

    if (!plans_a.empty()) {
      if (plans_a.front().cost > plan_a.cost) {
        plans_a.clear();
      } else if (plans_a.front().cost < plan_a.cost) {
        continue;
      } else if (plans_a.front().cost == plan_a.cost) {
        bool redundant = false;
        for (auto& other_plan : plans_a) {
          if (plan_a.SameExtractors(other_plan)) {
            redundant = true;
          }
        }
        if (redundant) {
          continue;
        }
      }
    }
    plans_a.push_back(plan_a);

    for (Number value_b = 1; value_b < N; ++value_b) {
      auto& plans_b = plans[value_b];
      if (plans_b.empty()) continue;

      auto Consider = [&](auto op) {
        auto validity = op.Valid(value_a, value_b);
        if (validity == Validity::Invalid) {
          return;
        }
        bool reverse = validity == Validity::Reverse;
        auto new_value = reverse ? op.Apply(value_b, value_a) : op.Apply(value_a, value_b);
        auto& other_plans = plans[new_value];
        auto rough_cost_estimate = plan_a.ops + op.extra_ops;
        if (rough_cost_estimate > kMaxCost) {
          return;
        }
        if (!other_plans.empty() && other_plans.front().cost < rough_cost_estimate) {
          return;
        }
        for (auto& plan_b : plans_b) {
          // optimistic estimate, assuming extractors combine perfectly without increasing cost
          auto cost_estimate = plan_a.ops + plan_b.cost + op.extra_ops;
          if (cost_estimate > kMaxCost) {
            continue;
          }
          if (!other_plans.empty() && other_plans.front().cost < cost_estimate) {
            continue;
          }
          auto new_plan = op.Combine(reverse ? plan_b : plan_a, reverse ? plan_a : plan_b);
          if (!other_plans.empty() && other_plans.front().cost < new_plan.cost) {
            continue;
          }
          q.push_back(new_plan);
        }
      };

      Consider(AddOp{});
      Consider(MulOp{});
      Consider(SubOp{});
      Consider(DivPlusRemOp{});
      Consider(DivMulRemOp{});
      Consider(DivSubRemOp{});
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