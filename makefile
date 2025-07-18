FILES = bin/framework_embedder

# Change this to your path for the TENNLab open source neuromorphic framework
# Or just set the fr_open environment variable to the open source framework path
framework = $(fr_open)

CFLAGS = -Wall -Wextra -std=c++11 -I$(framework)/include -I./include/
FLIB = $(framework)/lib/libframework.a
SRC = src/*.cpp
INC = include/*.hpp

C_OBJ = $(framework)/obj/framework.o $(framework)/obj/risp.o $(framework)/obj/risp_static.o

all: $(FILES)

clean:
	rm -f $(FILES)

bin/framework_embedder: $(SRC) $(INC)
	( mkdir bin ; cd $(framework) ; make )
	g++ -o bin/framework_embedder $(CFLAGS) $(SRC) $(C_OBJ) $(FLIB)
