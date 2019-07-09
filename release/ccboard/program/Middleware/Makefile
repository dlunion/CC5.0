ECHO = @echo
OUTNAME = ../../VisualizerMiddleware.so
CC := g++ -shared -fPIC
CUCC := nvcc

CFLAGS := -m64 -std=c++11 -g -openmp -O3 -fpermissive -fPIC
OPENCV_INC_PATH		 = -I/usr/include/opencv/ -I/usr/lib/x86_64-linux-gnu/ -Isrc -Iccinclude -I/datav/wish/9.cc5.0/3rd/boost/libs/include -I/root/anaconda3/include/python3.7m
INCLUDES := -I/usr/local/include $(OPENCV_INC_PATH)
CFLAGS += $(INCLUDES) -fPIC
LDPATH := -L/usr/lib -L/usr/local/lib -L./ -L/datav/wish/9.cc5.0/3rd/boost/libs/lib -L/root/anaconda3/lib

LDFLAGS :=  $(LDPATH) -ldl
RPATH   := -Wl,-rpath .
RPATH   += -Wl,-rpath /datav/wish/9.cc5.0/3rd/boost/libs/lib
RPATH   += -Wl,-rpath /root/anaconda3/lib
LDFLAGS += -ldl -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_filesystem -lboost_system -lboost_locale -lpthread -lpython3.7m -lboost_python37 $(RPATH)
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
