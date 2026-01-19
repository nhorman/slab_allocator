#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/queue.h>
#include <openssl/crypto.h>


static void *slab_malloc(size_t num, const char *file, int line)
{
    return malloc(num);
}

static void *slab_realloc(void *addr, size_t num, const char *file, int line)
{
    return realloc(addr, num);
}

static void slab_free(void *addr, const char *file, int line)
{
    free(addr);
}

static __attribute__((constructor)) void setup_slab_allocator()
{
    fprintf(stderr, "Setting up slab allocator\n");
    if (!CRYPTO_set_mem_functions(slab_malloc, slab_realloc, slab_free))
        fprintf(stderr, "Failed to setup slag allocator\n");
}

