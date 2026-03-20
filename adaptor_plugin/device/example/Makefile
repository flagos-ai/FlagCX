INCLUDE_DIRS = .

CXXFLAGS = $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) \
           -fPIC -fvisibility=hidden -Wall -g

LDFLAGS = -shared

build: libflagcx-device-example.so

libflagcx-device-example.so: plugin.cc
	g++ $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f libflagcx-device-example.so
