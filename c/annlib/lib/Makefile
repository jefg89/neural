# Makefile for annlib build

Phony: build 

#source macros
LIB=ann
SRC=$(LIB).c
INC=$(LIB).h
OBJ=$(LIB).o


#Toolchain macros
CXX=gcc
AR=ar
RANLIB=ranlib
LD=ld
CFLAGS= -fPIC -I../include
LDFLAGS = -Bdynamic -shared -lm

#Build rule

build: $(SRC) ../include/$(INC)
		$(CXX) -c -o $(OBJ) $(SRC) $(CFLAGS)
		$(LD)  -o lib$(LIB).so $(OBJ) $(LDFLAGS)



clean:
		@rm -rf *.o 
