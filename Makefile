CPP = g++

FLAGS = -Ofast -flto -march=native -funroll-loops -Wall -lm -pthread -Wno-unused-result
FLAGS += -Wno-c++11-extensions


all: main

main : main.cpp
	$(CPP) main.cpp -o main $(FLAGS)

clean: 
	rm main
