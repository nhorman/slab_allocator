all: slab_allocator.so

slab_allocator.so: slab_allocator.o
	$(CC) -shared -fPIC -o slab_allocator.so slab_allocator.o -L/home/nhorman/git/openssl -lcrypto


%.o: %.c
	$(CC) -fPIC -I/home/nhorman/git/openssl/include -c $< -o $@

clean:
	rm -f *.o *.so
