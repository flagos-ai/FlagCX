build: libflagcx-ccl-example.so

libflagcx-ccl-example.so: plugin.cc
	g++ -Iflagcx -fPIC -shared -o $@ $^

clean:
	rm -f libflagcx-ccl-example.so
