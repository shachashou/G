#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gomoku {
namespace json {

// Minimal JSON value tree for the executor's line protocol. Requests may
// contain objects, arrays, strings, numbers, bools and null.
class Value {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0;
  std::string string;
  std::vector<Value> array;
  std::vector<std::pair<std::string, Value>> object;

  const Value* find(const std::string& key) const;
  bool isArray() const { return type == Type::Array; }
  bool isObject() const { return type == Type::Object; }
  std::string asString(const std::string& fallback = {}) const;
  long long asInt(long long fallback = 0) const;
  bool asBool(bool fallback = false) const;
};

class ParseError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Parses exactly one JSON value; trailing content is an error.
Value parse(const std::string& text);

// Builders for responses.
Value nullValue();
Value booleanValue(bool v);
Value numberValue(long long v);
Value stringValue(std::string v);
Value arrayValue(std::vector<Value> items = {});
Value objectValue();
void set(Value& object, const std::string& key, Value v);
std::string dump(const Value& value);

}  // namespace json
}  // namespace gomoku
