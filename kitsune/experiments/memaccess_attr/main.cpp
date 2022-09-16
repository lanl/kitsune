#include <iostream>
#include <kitsune.h>

_writeonly int myfunc(_readonly int a, _readonly int b, _readwrite int* c) {
  int i;

  *c = a + b;
  i = *c;
  return i;
}

int main (int argc, char** argv) {
  int i,j,k;
  i=1;
  j=1;
  k=0;

  myfunc(i,j,&k);
  std::cout << "i = " << i << std::endl;
  std::cout << "j = " << j << std::endl;
  std::cout << "k = " << k << std::endl;
  return 0;
}
