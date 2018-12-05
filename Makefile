# Makefile for cub-test
CC			=	nvcc
NVCC		=	nvcc
LD			=	nvcc

EXE			:= 	redsync-test
#CPP_SRCS := kernels.cc
#CPP_OBJS := ${CPP_SRCS:.cc=.o}
CU_SRCS	:= main.cu
CU_OBJS	:=	${CU_SRCS:.cu=.o}
OBJS	:=	$(CU_OBJS) #$(CPP_OBJS) $(CU_OBJS)

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

.PHONY:		all clean distclean

$(EXE): $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -o $(EXE)

.cc.o:
	$(CC) $(CCFLAGS) $<

%.o : %.cu
	$(NVCC) $(CCFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	rm -vf $(EXE) $(OBJS)
