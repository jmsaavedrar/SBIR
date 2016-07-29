#it should be changed to the current SBIR directory
SBIR_HOME := /home/jsaavedr/Research/git/SBIR
#
JMSR_DIR := $(SBIR_HOME)/OrandLibs/jmsr
CVISION_DIR := $(SBIR_HOME)/OrandLibs/cvision
#
CXX=g++
LINK :=$(CXX)
CXXFLAGS :=-m64 -std=c++11 -O2 -Wall -W -Wno-unknown-pragmas
PKG_DEPS := opencv flann
INCPATH :=$(shell pkg-config --cflags  $(PKG_DEPS))
INCPATH +=-I$(JMSR_DIR)/include
INCPATH +=-I$(CVISION_DIR)/include

LIBS :=$(shell pkg-config --libs $(PKG_DEPS))
LIBS +=-L$(JMSR_DIR)/lib -ljmsr
LIBS +=-L$(CVISION_DIR)/lib -lcvision

LFLAGS :=-m64 -Wl,-O1 
LFLAGS +=-Wl,-rpath,$(JMSR_DIR)/lib
LFLAGS +=-Wl,-rpath,$(CVISION_DIR)/lib

SRC_DIR :=src
OBJ_DIR :=obj

TARGET :=LKS
CURDIR :=$(shell pwd)
all: dep $(OBJ_DIR) $(TARGET)
dep:
	cd $(JMSR_DIR) && make && cd $(CURDIR)
	cd $(CVISION_DIR) && make && cd $(CURDIR)
	
$(OBJ_DIR):
	mkdir -p $@

$(TARGET): $(OBJ_DIR)/getLKSDescriptor.o
	$(LINK) $(LFLAGS) -o "$@"  "$^" $(LIBS)

$(OBJ_DIR)/%.o:$(SRC_DIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

clean:
	cd $(JMSR_DIR) && make clean && cd $(CURDIR)
	cd $(CVISION_DIR) && make clean && cd $(CURDIR)
	rm $(OBJ_DIR)/* $(TARGET) 
