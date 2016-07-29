#Written by Jose M. Saavedra
CC := g++
LIBDIR=lib
LIB_NAME := jmsr
TARGET := $(LIBDIR)/lib$(LIB_NAME).so
PK_DEPS := opencv flann
CFLAGS	= -Wall -Wextra -pedantic -fpic -Wunknown-pragmas -fopenmp  -std=c++11 -DNO_VLFEAT
LD_FLAGS += -Wall -shared
INCPATH :=  -Iinclude -Iinclude/$(LIB_NAME)
INCPATH += $(shell pkg-config --cflags $(PK_DEPS))
LIBS := $(shell pkg-config --libs $(PK_DEPS))

SOURCE  :=  $(shell ls  src/$(LIB_NAME)/*.cpp)
HEADERS := $(shell ls include/$(LIB_NAME)/*.[h,hpp])
SOURCE := $(notdir $(SOURCE))
OBJDIR=obj
OBJECTS := $(addprefix $(OBJDIR)/,$(SOURCE:.cpp=.o)) 

all: $(LIBDIR) $(OBJDIR) $(TARGET)

$(OBJDIR):
	mkdir -p $@	

$(LIBDIR):
	mkdir -p $@	

$(TARGET): $(OBJECTS)
	$(CC) $(LD_FLAGS) $^ -o $@ $(LIBS)


$(OBJDIR)/%.o: src/$(LIB_NAME)/%.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $^ -o $@ 

clean :
	rm -f $(TARGET) $(OBJECTS)

