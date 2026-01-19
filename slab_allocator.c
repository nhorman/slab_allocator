#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>
#include <sys/queue.h>
#include <openssl/crypto.h>

static long page_size = 0;

struct slab_info;

struct slab_ring {
    LIST_ENTRY(slab_ring) entry; 
    struct slab_info *info;
    uint32_t available_objs;
    uint64_t *bitmap;
    uint32_t bitmap_word_count;
};

LIST_HEAD(slab_entries, slab_ring);

struct slab_info {
    pthread_rwlock_t ring_lock;
    struct slab_entries entries;
    size_t obj_size;
};

#define MAX_SLAB 1 << 10
static struct slab_info slabs[] = {
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), 1 << 0},
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), 1 << 1},
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), 1 << 2},
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), 1 << 3},
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), 1 << 4},
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), 1 << 5},
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), 1 << 6},
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), 1 << 7},
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), 1 << 8},
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), 1 << 9},
    {PTHREAD_RWLOCK_INITIALIZER, LIST_HEAD_INITIALIZER(entries), MAX_SLAB}
};

static unsigned int get_slab_idx(size_t num)
{
    if (num == 0)
        return 0; /* return index zero */

    /* round up to the next power of two */
    num--;
    num |= num >> 1;
    num |= num >> 2;
    num |= num >> 4;
    num |= num >> 8;
    num |= num >> 16;
    num |= num >> 32;
    return __builtin_ctzl(num);
}

#define PAGE_MASK (~(page_size - 1))
#define PAGE_START(x) (void *)((uintptr_t)(x) & PAGE_MASK)

static inline struct slab_ring *get_slab_ring(void *addr)
{
    uintptr_t slab_ring_ptr = (uintptr_t)PAGE_START(addr);
    slab_ring_ptr += page_size;
    slab_ring_ptr -= sizeof(struct slab_ring);
    return (struct slab_ring *)slab_ring_ptr;
}

static void *select_obj(struct slab_ring *slab)
{
    uint32_t i;
    uint64_t value;
    uint32_t available_bit;
    uint64_t new_mask;
    void *slab_start = PAGE_START(slab);
    uint32_t obj_offset;
    void *obj;

    for (i=0;i<slab->bitmap_word_count;i++) {
try_again:
        value = __atomic_load_n(&slab->bitmap[i], __ATOMIC_RELAXED);
        if (value < UINT64_MAX) {
            value = ~slab->bitmap[i];
            available_bit = __builtin_ctzl(value);
            new_mask = 1 << available_bit;
            value = atomic_fetch_or(&slab->bitmap[i], new_mask);
            if ((value & new_mask) == new_mask) {
                /* another  thread already set this bit, try again */
                goto try_again;
            }
            /* We got an object! */
            obj_offset = (slab->info->obj_size * (i * 64)) + (available_bit * slab->info->obj_size);
            return (void *)((unsigned char *)slab_start + obj_offset);
        }
    }
    return NULL;
}

static struct slab_ring *create_new_slab(struct slab_info *slab)
{
    void *new;
    size_t page_size_long = (size_t)page_size;
    struct slab_ring *new_ring;
    uint32_t count;
    size_t computed_size;
    size_t available_size;
    uintptr_t bitmap_ptr;

    if (!posix_memalign(&new, page_size_long, page_size_long))
        return NULL;

    /*
     * setup this slab
     */
    new_ring = get_slab_ring(new);

    new_ring->info = slab;
    new_ring->available_objs = 0;
    available_size = (page_size - sizeof(struct slab_ring)) * slab->obj_size;
    for (count = 0; count < (page_size - sizeof(struct slab_ring)) * slab->obj_size; count+=64) {
        computed_size = (count * sizeof(slab->obj_size));
        if (computed_size > available_size)
            break;
        available_size -= sizeof(uint64_t);
    }
    new_ring->available_objs = count - 64;
    bitmap_ptr = (uintptr_t)new_ring;
    bitmap_ptr -= count / 64;
    new_ring->bitmap = (uint64_t *)bitmap_ptr;
    return new_ring;
}

static void *create_obj_in_new_slab(struct slab_info *slab)
{
    struct slab_ring *new = create_new_slab(slab);
    void *obj;

    if (new == NULL)
        return NULL;
    /*
     * we can cheat here a bit, since no one else sees this slab yet
     */
    obj = PAGE_START(new);
    new->available_objs--;
    new->bitmap[0] = 0x1;

    /*
     * Now insert the slab to the list
     */
    pthread_rwlock_wrlock(&slab->ring_lock);
    LIST_INSERT_HEAD(&slab->entries, new, entry);
    pthread_rwlock_unlock(&slab->ring_lock);
    return obj;
}

static void *get_slab_obj(struct slab_info *slab)
{
    struct slab_ring *idx;
    void *obj;
    uint32_t available, new;

    pthread_rwlock_rdlock(&slab->ring_lock);
    LIST_FOREACH(idx, &slab->entries, entry) {
try_again:
        available = __atomic_load_n(&idx->available_objs, __ATOMIC_RELAXED);
        if (available > 0) {
            new = available - 1;
            if (!__atomic_compare_exchange(&idx->available_objs, &available, &new, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                goto try_again;
            /* There is at least one free object in this slab */
            obj = select_obj(idx);
            pthread_rwlock_unlock(&slab->ring_lock);
            return obj;
        }
    }
    pthread_rwlock_unlock(&slab->ring_lock);
    /* We need to create a new slab */
    return create_obj_in_new_slab(slab);
}

static void *slab_malloc(size_t num, const char *file, int line)
{
    unsigned int slab_idx;

    if (num > MAX_SLAB)
        return malloc(num);

    slab_idx = get_slab_idx(num);
    return get_slab_obj(&slabs[slab_idx]);
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
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1)
        fprintf(stderr, "Failed to get page size\n");
}

