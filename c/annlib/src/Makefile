# Makefile for test compile

Phony: all
#source macros
APP=ann
SRC=$(APP)_test.c
INC=$(APP).h
OBJ=$(APP).o


#Toolchain macros
CXX=gcc
CFLAGS=  -I../include
LDFLAGS = -L ../lib -lm -lann

#Build rule

all: $(SRC) ../include/$(INC)
		$(CXX) -c -o  $(OBJ) $(SRC) $(CFLAGS) 
		$(CXX) -o ../bin/$(APP) $(OBJ) $(LDFLAGS)
		@echo ----My work here is done----

clean:
		@rm -rf $(APP) ../bin/$(APP) *.o 
