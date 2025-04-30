#include "debug.hpp"
#include <iostream>
namespace Artyr99M {
#ifdef NDEBUG
std::ostream& debug = std::cerr;
#else
debuging& debuging::operator<<(std::basic_ostream<char, std::char_traits<char>>& (*func)(std::basic_ostream<char, std::char_traits<char>>&)) {
    return *this;
}
debuging& debuging::flush() {
    return *this;
}
debuging debug;
#endif //DEBUG
} //namespace Artyr99M
