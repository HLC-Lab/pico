CXXFLAGS=-g -O3 -Wno-deprecated -Wall
CCFLAGS=-g -O3 -g
LDFLAGS=-g -O3 -g #-lboost_iostreams-mt 

# Directory structure
OBJDIR=obj
BINDIR=bin

# Object file lists with obj/ prefix
COLL_OBJS=$(OBJDIR)/coll/bcast.o \
          $(OBJDIR)/coll/reduce.o \
          $(OBJDIR)/coll/barrier.o \
          $(OBJDIR)/coll/allreduce.o \
          $(OBJDIR)/coll/alltoall.o \
          $(OBJDIR)/coll/gather.o \
          $(OBJDIR)/coll/scatter.o \
          $(OBJDIR)/coll/pairwise.o \
          $(OBJDIR)/coll/collective_registry.o \
          $(OBJDIR)/coll/collective_selector.o

OBJS=$(OBJDIR)/buffer_element.o \
     $(OBJDIR)/cmdline.o \
     $(OBJDIR)/process_trace.o \
     $(OBJDIR)/schedgen.o \
     $(COLL_OBJS)

force: all

# Create directories if they don't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/coll:
	mkdir -p $(OBJDIR)/coll

$(BINDIR):
	mkdir -p $(BINDIR)

cmdline.c: schedgen.ggo
	gengetopt -i schedgen.ggo

$(OBJDIR)/%.o: %.cpp *hpp *h | $(OBJDIR)
	g++ $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.c *h | $(OBJDIR)
	gcc $(CCFLAGS) -c $< -o $@

$(OBJDIR)/coll/%.o: coll/%.cpp | $(OBJDIR)/coll
	g++ $(CXXFLAGS) -c $< -o $@

all: $(OBJS) schedgen.ggo | $(BINDIR)
	g++ $(CXXFLAGS) $(OBJS) -o $(BINDIR)/schedgen $(LDFLAGS)

clean:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)
	rm -f cmdline.c cmdline.h

.PHONY: force all clean
