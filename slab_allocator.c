#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/queue.h>
#include <openssl/crypto.h>


static __attribute__((constructor)) void setup_slab_allocator()
{
    fprintf(stderr, "Setting up slab allocator\n");
}

