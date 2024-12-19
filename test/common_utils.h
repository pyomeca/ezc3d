#ifndef __TEST_COMMON_UTILS_H__
#define __TEST_COMMON_UTILS_H__

#define EXPECT_DOUBLE_AS_FLOAT_EQ(a, b)                                        \
  EXPECT_FLOAT_EQ(static_cast<float>(a), static_cast<float>(b))

#endif // __TEST_COMMON_UTILS_H__