// RUN: %kitxx -Xclang -verify -fsyntax-only -ftapir=cuda %s

#include <kitsune.h>

int main(int argc, char *argv[]) {
  // expected-error@+1 {{launch attribute: threads-per-block must be a positive integer value}}
  [[kitsune::launch(-1)]]
  forall(int i = 0; i < 1024; ++i) { }

  // expected-error@+1 {{'launch' attribute takes one argument}}
  [[kitsune::launch()]]
  forall(int i = 0; i < 1024; ++i) { }

  // expected-error@+1 {{'launch' attribute takes one argument}}
  [[kitsune::launch(32, 64)]]
  forall(int i = 0; i < 1024; ++i) { }

  // expected-error@+1 {{launch attribute: threads-per-block must be a built-in integer type}}
  [[kitsune::launch(1 + 2.3)]]
  forall(int i = 0; i < 1024; ++i) { }

  // expected-error@+1 {{launch attribute: threads-per-block must be a built-in integer type}}
  [[kitsune::launch("32")]]
  forall(int i = 0; i < 1024; ++i) { }

  // expected-error@+1 {{launch attribute: threads-per-block must be a built-in integer type}}
  [[kitsune::launch(2.3)]]
  forall(int i = 0; i < 1024; ++i) { }

  // expected-error@+1 {{'launch' attribute only applies to 'forall' statement}}
  [[kitsune::launch(32)]]
  spawn s {}

  // expected-error@+1 {{'launch' attribute only applies to 'forall' statement}}
  [[kitsune::launch(45)]]
  if (argc == 1) {
    forall(int i = 0; i < 1024; ++i) { }
  }

  return 0;
}

