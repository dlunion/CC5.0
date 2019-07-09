ECHO = @echo
OUTNAME = ../../ccboard
CC := g++ -fPIC
CUCC := nvcc

CFLAGS := -m64 -std=c++11 -g -openmp -O3 -fpermissive -fPIC
OPENCV_INC_PATH		 = -I/usr/lib/x86_64-linux-gnu/
INCLUDES := -I/usr/local/include $(OPENCV_INC_PATH)
CFLAGS += $(INCLUDES) -fPIC
LDPATH := -L/usr/lib -L/usr/local/lib -L./

LDFLAGS :=  $(LDPATH) -ldl
RPATH   := -Wl,-rpath .
RPATH   += -Wl,-rpath /datav/wish/9.cc5.0/3rd/boost/libs/lib
RPATH   += -Wl,-rpath /root/anaconda3/lib
LDFLAGS += -ldl $(RPATH)
SRCDIR := ./
OBJDIR := ./objs
BINDIR := ./

OBJS = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(wildcard $(SRCDIR)/*.cpp))

all: $(BINDIR)/$(OUTNAME)
#run: $(BINDIR)/$(OUTNAME)
#cd $(BINDIR) && ./$(OUTNAME)

$(BINDIR)/$(OUTNAME): $(OBJS) $(CUOBJS)
	$(ECHO) Linking: $@
	$(CC) -o $@ $^ $(LDFLAGS)
	#$(BINDIR)/$(OUTNAME)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(CUCC) $(CUFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)/$(OUTNAME)
