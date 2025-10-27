#ifndef __TEST_COMMON_UTILS_H__
#define __TEST_COMMON_UTILS_H__

#define EXPECT_DOUBLE_AS_FLOAT_EQ(a, b)                                        \
  EXPECT_FLOAT_EQ(static_cast<float>(a), static_cast<float>(b))

// Trim from the start (in place)
inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

// Trim from the end (in place)
inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// Trim from the start (copying)
inline std::string trim(std::string s) {
  rtrim(s);
  ltrim(s);
  return s;
}

#endif // __TEST_COMMON_UTILS_H__