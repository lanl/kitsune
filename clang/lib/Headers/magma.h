#include<string>
#include<limits>

template <typename a> 
struct Magma {
  virtual a op(a x, a y) = 0; 
}; 

template <typename a>
struct UnitalMagma : public Magma<a> {
  virtual a id() = 0; 
}; 

// Example unital magmas
template <typename a>
struct Sum : UnitalMagma<a>{
  a op(a x, a y){ return x + y; }
  a id(){ return 0; }  // look into this more
}; 

template <typename a>
struct Product : UnitalMagma<a> {
  a op(a x, a y){ return x * y; }
  a id(){ return 1; }
}; 

struct StringApp : UnitalMagma<std::string> {
  std::string op(std::string x, std::string y){ return x.append(y); }
  std::string id() { return ""; }
}; 

template <typename a>
struct Max : UnitalMagma<a> {
  a op(a x, a y){ return x > y ? x : y; }
  a id() { return std::numeric_limits<a>::min(); }
}; 

template <typename a>
struct Min : UnitalMagma<a> {
  a op(a x, a y){ return x < y ? x : y; }
  a id() { return std::numeric_limits<a>::max(); }
}; 

