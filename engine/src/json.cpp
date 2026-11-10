#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace gomoku {
namespace json {

const Value* Value::find(const std::string& key) const {
  if (type != Type::Object) return nullptr;
  for (const auto& kv : object) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

std::string Value::asString(const std::string& fallback) const {
  return type == Type::String ? string : fallback;
}

long long Value::asInt(long long fallback) const {
  return type == Type::Number ? static_cast<long long>(number) : fallback;
}

bool Value::asBool(bool fallback) const {
  return type == Type::Bool ? boolean : fallback;
}

namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  Value run() {
    skip();
    Value value = parseValue();
    skip();
    if (pos_ != text_.size()) fail("trailing content");
    return value;
  }

 private:
  const std::string& text_;
  size_t pos_ = 0;

  [[noreturn]] void fail(const char* what) const {
    throw ParseError(std::string("json: ") + what + " at byte " +
                     std::to_string(pos_));
  }
  void skip() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
      ++pos_;
    }
  }
  char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }
  char take() { return pos_ < text_.size() ? text_[pos_++] : '\0'; }
  void expect(char c) {
    if (take() != c) fail("unexpected character");
  }
  void literal(const char* word) {
    const size_t n = std::strlen(word);
    if (text_.compare(pos_, n, word) != 0) fail("bad literal");
    pos_ += n;
  }

  Value parseValue() {
    skip();
    if (pos_ >= text_.size()) fail("unexpected end of input");
    switch (peek()) {
      case '{': return parseObject();
      case '[': return parseArray();
      case '"': {
        Value v;
        v.type = Value::Type::String;
        v.string = parseString();
        return v;
      }
      case 't': {
        literal("true");
        Value v;
        v.type = Value::Type::Bool;
        v.boolean = true;
        return v;
      }
      case 'f': {
        literal("false");
        Value v;
        v.type = Value::Type::Bool;
        v.boolean = false;
        return v;
      }
      case 'n':
        literal("null");
        return Value{};
      default:
        return parseNumber();
    }
  }

  Value parseObject() {
    Value v;
    v.type = Value::Type::Object;
    expect('{');
    skip();
    if (peek() == '}') {
      ++pos_;
      return v;
    }
    while (true) {
      skip();
      if (peek() != '"') fail("object key must be a string");
      std::string key = parseString();
      skip();
      expect(':');
      v.object.emplace_back(std::move(key), parseValue());
      skip();
      const char c = take();
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}'");
    }
    return v;
  }

  Value parseArray() {
    Value v;
    v.type = Value::Type::Array;
    expect('[');
    skip();
    if (peek() == ']') {
      ++pos_;
      return v;
    }
    while (true) {
      v.array.push_back(parseValue());
      skip();
      const char c = take();
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']'");
    }
    return v;
  }

  unsigned hex4() {
    unsigned value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = take();
      value <<= 4;
      if (c >= '0' && c <= '9') {
        value |= static_cast<unsigned>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        value |= static_cast<unsigned>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        value |= static_cast<unsigned>(c - 'A' + 10);
      } else {
        fail("bad \\u escape");
      }
    }
    return value;
  }

  static void appendUtf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  std::string parseString() {
    expect('"');
    std::string out;
    while (true) {
      const char c = take();
      if (c == '\0') fail("unterminated string");
      if (c == '"') break;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      const char e = take();
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF && peek() == '\\' &&
              pos_ + 1 < text_.size() && text_[pos_ + 1] == 'u') {
            take();
            take();
            const unsigned low = hex4();
            if (low >= 0xDC00 && low <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            }
          }
          appendUtf8(out, cp);
          break;
        }
        default: fail("bad string escape");
      }
    }
    return out;
  }

  Value parseNumber() {
    const size_t start = pos_;
    if (peek() == '-') ++pos_;
    if (peek() == '0') {
      ++pos_;
    } else {
      if (peek() < '1' || peek() > '9') fail("bad number");
      while (peek() >= '0' && peek() <= '9') ++pos_;
    }
    if (peek() == '.') {
      ++pos_;
      if (peek() < '0' || peek() > '9') fail("bad fraction");
      while (peek() >= '0' && peek() <= '9') ++pos_;
    }
    if (peek() == 'e' || peek() == 'E') {
      ++pos_;
      if (peek() == '+' || peek() == '-') ++pos_;
      if (peek() < '0' || peek() > '9') fail("bad exponent");
      while (peek() >= '0' && peek() <= '9') ++pos_;
    }
    Value v;
    v.type = Value::Type::Number;
    v.number = std::strtod(text_.substr(start, pos_ - start).c_str(), nullptr);
    return v;
  }
};

void dumpString(const std::string& s, std::string& out) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

void dumpInto(const Value& v, std::string& out) {
  switch (v.type) {
    case Value::Type::Null: out += "null"; break;
    case Value::Type::Bool: out += v.boolean ? "true" : "false"; break;
    case Value::Type::Number:
      if (v.number == std::floor(v.number) && v.number >= -9e18 &&
          v.number <= 9e18) {
        out += std::to_string(static_cast<long long>(v.number));
      } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.17g", v.number);
        out += buf;
      }
      break;
    case Value::Type::String: dumpString(v.string, out); break;
    case Value::Type::Array: {
      out.push_back('[');
      for (size_t i = 0; i < v.array.size(); ++i) {
        if (i) out.push_back(',');
        dumpInto(v.array[i], out);
      }
      out.push_back(']');
      break;
    }
    case Value::Type::Object: {
      out.push_back('{');
      for (size_t i = 0; i < v.object.size(); ++i) {
        if (i) out.push_back(',');
        dumpString(v.object[i].first, out);
        out.push_back(':');
        dumpInto(v.object[i].second, out);
      }
      out.push_back('}');
      break;
    }
  }
}

}  // namespace

Value parse(const std::string& text) { return Parser(text).run(); }

Value nullValue() { return Value{}; }

Value booleanValue(bool v) {
  Value out;
  out.type = Value::Type::Bool;
  out.boolean = v;
  return out;
}

Value numberValue(long long v) {
  Value out;
  out.type = Value::Type::Number;
  out.number = static_cast<double>(v);
  return out;
}

Value stringValue(std::string v) {
  Value out;
  out.type = Value::Type::String;
  out.string = std::move(v);
  return out;
}

Value arrayValue(std::vector<Value> items) {
  Value out;
  out.type = Value::Type::Array;
  out.array = std::move(items);
  return out;
}

Value objectValue() {
  Value out;
  out.type = Value::Type::Object;
  return out;
}

void set(Value& object, const std::string& key, Value v) {
  object.object.emplace_back(key, std::move(v));
}

std::string dump(const Value& value) {
  std::string out;
  dumpInto(value, out);
  return out;
}

}  // namespace json
}  // namespace gomoku
