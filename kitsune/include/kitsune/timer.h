#ifndef __KIT_TIMER_H__
#define __KIT_TIMER_H__

#include <chrono>

namespace kitsune
{
  class timer {
    public:
      inline timer() { reset(); }
      inline ~timer() = default;
      
      inline void reset() {
	using namespace std::chrono;
	stamp = steady_clock::now();
      };

      inline double seconds() const {
	using namespace std::chrono;
	steady_clock::time_point now;
	now = steady_clock::now();
	return duration_cast<duration<double>>(now - stamp).count();
      }

    private:
      timer(const timer&);
      timer& operator=(const timer&);
      std::chrono::steady_clock::time_point stamp;
  };
}

#endif

