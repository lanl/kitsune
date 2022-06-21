
cc=clang
cxx=clang++

all: extern

add.o: add.c
	${cc} -c -flto $< -O1 -o $@

extern.o: extern.cpp
	${cxx} -c -fno-exceptions -flto $< -o $@

extern: add.o extern.o
	${cxx} -fuse-ld=lld -flto $^ -ftapir=gpu -O1 -o  $@ 

clean: 
	rm -f add.o extern.o extern
