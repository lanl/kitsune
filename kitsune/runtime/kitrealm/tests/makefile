cflags=-lrealm -lpthread -ldl -lrt -g

test: test.cpp realm_wrapper.o
	g++ test.cpp realm_wrapper.o ${cflags}  -o test

realm_wrapper.o: realm_wrapper.cpp
	g++ -g -c realm_wrapper.cpp 

