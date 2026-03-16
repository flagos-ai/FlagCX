build: libflagcx-net-example.so

libflagcx-net-example.so: plugin.cc
	g++ -Iflagcx -fPIC -shared -o $@ $^

clean:
	rm -f libflagcx-net-example.so
