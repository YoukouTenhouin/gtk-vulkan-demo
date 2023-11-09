CFLAGS=-Wall -Wextra `pkg-config --cflags gtkmm-4.0 vulkan` -std=c++20
LDFLAGS=`pkg-config --libs gtkmm-4.0 vulkan`
CXX=g++
GLSL=glslc
BFD=ld.bfd

.PHONY: clean all

all: gtk-vulkan-daemon

gtk-vulkan-daemon: gtk_vulkan_demo.o shader_vert.o shader_frag.o
	$(CXX) gtk_vulkan_demo.o shader_vert.o shader_frag.o -ogtk-vulkan-daemon $(CFLAGS) $(LDFLAGS)

gtk_vulkan_demo.o: gtk_vulkan_demo.cc
	$(CXX) -c -o gtk_vulkan_demo.o gtk_vulkan_demo.cc $(CFLAGS)

shader_vert.o: shader.vert
	$(GLSL) shader.vert -o shader_vert.spv
	$(BFD) -r -b binary -o shader_vert.o shader_vert.spv

shader_frag.o: shader.frag
	$(GLSL) shader.frag -o shader_frag.spv
	$(BFD) -r -b binary -o shader_frag.o shader_frag.spv

clean:
	rm -f *.o *.spv gtk-vulkan-daemon
