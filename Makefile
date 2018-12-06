# Makefile for cub-test
CC			=	nvcc
NVCC		=	nvcc
LD			=	nvcc

EXE			:= 	redsync
TEST		:=  trimmed-topk-test
#CPP_SRCS := kernels.cc
#CPP_OBJS := ${CPP_SRCS:.cc=.o}
CU_SRCS	:= main.cu
CU_TEST_SRCS := test_trimmed_topk.cu
CU_OBJS	:=	${CU_SRCS:.cu=.o}
CU_TEST_OBJS	:=	${CU_TEST_SRCS:.cu=.o}
OBJS	:=	$(CU_OBJS)
TEST_OBJS := $(CU_TEST_OBJS)

# include directories
INC_DIRS	:=	/usr/local/cuda/include \
                        ./cub

# library directories
LIB_DIRS	:=	/usr/local/cuda/lib64

# libraries
LIBS		:=	cudart

CCFLAGS		+=	-c -g $(foreach includedir, $(INC_DIRS), -I $(includedir))
LDFLAGS		+=	$(foreach librarydir, $(LIB_DIRS), -L $(librarydir))
LDFLAGS		+=	$(foreach library, $(LIBS), -l$(library))

$(CU_OBJS): EXTRA_FLAGS := -arch compute_61
$(CU_TEST_OBJS): EXTRA_FLAGS := -arch compute_61 -std=c++11

.PHONY:		all clean distclean

$(EXE): $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -o $(EXE)

$(TEST): $(TEST_OBJS)
	$(LD) $(TEST_OBJS) $(LDFLAGS) -o $(TEST)

.cc.o:
	$(CC) $(CCFLAGS) $<

%.o : %.cu
	$(NVCC) $(CCFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	rm -vf $(EXE) $(OBJS) $(TEST) $(TEST_OBJS)
