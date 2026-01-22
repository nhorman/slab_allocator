#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <stdatomic.h>
#include <sys/queue.h>
#include <sys/mman.h>
#include <openssl/crypto.h>

static long page_size = 0;

struct slab_info;

#define SLAB_MAGIC 0xdeadf00ddeadf00dUL

#ifdef SLAB_STATS
struct slab_stats {
    size_t allocs;
    size_t frees;
    size_t slabs;
};

#define INC_SLAB_STAT(metric) __atomic_add_fetch(metric, 1, __ATOMIC_ACQ_REL)
#else
#define INC_SLAB_STAT(metric)
#endif

struct slab_ring {
    LIST_ENTRY(slab_ring) entry; 
    struct slab_info *info;
    uint32_t available_objs;
    uint64_t *bitmap;
    uint32_t bitmap_word_count;
    void *obj_start;
    uint64_t magic;
};

LIST_HEAD(slab_entries, slab_ring);

struct slab_template {
    uint32_t available_objs;
    uint32_t bitmap_word_count;
    uint64_t last_word_mask;
};

struct slab_info {
    pthread_rwlock_t ring_lock;
    struct slab_ring *available;
    struct slab_entries entries;
    size_t obj_size;
#ifdef SLAB_STATS
    struct slab_stats stats;
#endif
    struct slab_template template;
};

#define MAX_SLAB_IDX 10
#define MAX_SLAB 1 << MAX_SLAB_IDX
#define EMPTY_SLAB_TEMPLATE { 0, 0 }

static struct slab_info slabs[] = {
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), 1 << 0, EMPTY_SLAB_TEMPLATE},
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), 1 << 1, EMPTY_SLAB_TEMPLATE},
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), 1 << 2, EMPTY_SLAB_TEMPLATE},
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), 1 << 3, EMPTY_SLAB_TEMPLATE},
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), 1 << 4, EMPTY_SLAB_TEMPLATE},
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), 1 << 5, EMPTY_SLAB_TEMPLATE},
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), 1 << 6, EMPTY_SLAB_TEMPLATE},
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), 1 << 7, EMPTY_SLAB_TEMPLATE},
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), 1 << 8, EMPTY_SLAB_TEMPLATE},
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), 1 << 9, EMPTY_SLAB_TEMPLATE},
    {PTHREAD_RWLOCK_INITIALIZER, NULL, LIST_HEAD_INITIALIZER(entries), MAX_SLAB, EMPTY_SLAB_TEMPLATE}
};

static inline size_t slab_size(size_t num)
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
    num++;
    return num;
}

static unsigned int get_slab_idx(size_t num)
{
    size_t up_size = slab_size(num);

    return __builtin_ctzl(up_size);
}

#define PAGE_MASK (~(page_size - 1))
#define PAGE_START(x) (void *)((uintptr_t)(x) & PAGE_MASK)

static inline struct slab_ring *get_slab_ring(void *addr)
{
    return (struct slab_ring *)PAGE_START(addr);
}

static inline int is_obj_slab(void *addr)
{
    struct slab_ring *slab = get_slab_ring(addr);

    return (slab->magic == SLAB_MAGIC) ? 1 : 0;
}

static void *select_obj(struct slab_ring *slab)
{
    uint32_t i;
    uint64_t value;
    uint32_t available_bit;
    uint64_t new_mask;
    uint32_t obj_offset;
    void *obj;

    for (i=0;i<slab->bitmap_word_count;i++) {
try_again:
        value = __atomic_load_n(&slab->bitmap[i], __ATOMIC_RELAXED);
        if (value < UINT64_MAX) {
            value = ~slab->bitmap[i];
            available_bit = __builtin_ctzl(value);
            new_mask = (uint64_t)1 << available_bit;
            value = atomic_fetch_or(&slab->bitmap[i], new_mask);
            if ((value & new_mask) == new_mask) {
                /* another  thread already set this bit, try again */
                goto try_again;
            }
            /* We got an object! */
            obj_offset = (slab->info->obj_size * (i * 64)) + (available_bit * slab->info->obj_size);
            return (void *)((unsigned char *)slab->obj_start + obj_offset);
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

    new = mmap(NULL, page_size_long, PROT_READ|PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (new == NULL)
        return NULL;

    /*
     * setup this slab
     */
    new_ring = get_slab_ring(new);

    new_ring->info = slab;
    new_ring->available_objs = slab->template.available_objs;
    new_ring->bitmap_word_count = slab->template.bitmap_word_count;
    new_ring->bitmap = (uint64_t *)(((unsigned char *)new_ring) + sizeof(struct slab_ring));
    memset(new_ring->bitmap, 0, sizeof(uint64_t) * new_ring->bitmap_word_count);
    new_ring->bitmap[new_ring->bitmap_word_count - 1] = slab->template.last_word_mask;
    new_ring->obj_start = (void *)(new_ring->bitmap + new_ring->bitmap_word_count);
    new_ring->magic = SLAB_MAGIC;
    INC_SLAB_STAT(&slab->stats.slabs);
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
    new->bitmap[0] |= 0x1;
    obj = new->obj_start;

    /*
     * Now insert the slab to the list
     */
    pthread_rwlock_wrlock(&slab->ring_lock);
    LIST_INSERT_HEAD(&slab->entries, new, entry);
    pthread_rwlock_unlock(&slab->ring_lock);
    __atomic_store(&slab->available, &new, __ATOMIC_RELAXED);
    return obj;
}

static void *get_slab_obj(struct slab_info *slab)
{
    struct slab_ring *idx;
    void *obj = NULL;
    
    idx = __atomic_load_n(&slab->available, __ATOMIC_RELAXED);
    if (idx != NULL)
        obj = select_obj(idx);
    if (obj != NULL)
        return obj;

    pthread_rwlock_rdlock(&slab->ring_lock);
    LIST_FOREACH(idx, &slab->entries, entry) {
        obj = select_obj(idx);
        if (obj != NULL)
            break;
    }
    pthread_rwlock_unlock(&slab->ring_lock);
    if (obj != NULL)
        return obj;
    /* We need to create a new slab */
new_slab:
    return create_obj_in_new_slab(slab);
}

static void return_to_slab(void *addr, struct slab_ring *ring)
{
    uintptr_t base;
    uintptr_t offset;
    size_t bit_idx;
    size_t word_idx;
    uint64_t value;
    uint32_t available, new;

    base = (uintptr_t)ring->obj_start;
    offset = (uintptr_t)addr - base;
    bit_idx = offset / ring->info->obj_size;
    word_idx = bit_idx / 64;

    bit_idx = bit_idx % 64;

    value = (uint64_t)1 << bit_idx;
    value = ~value;

    /*
     * Clear the bit for this object
     */
    __atomic_and_fetch(&ring->bitmap[word_idx], value, __ATOMIC_RELAXED);
}

static void *slab_malloc(size_t num, const char *file, int line)
{
    unsigned int slab_idx;

    if (num > MAX_SLAB)
        return malloc(num);

    slab_idx = get_slab_idx(num);
    INC_SLAB_STAT(&slabs[slab_idx].stats.allocs);
    return get_slab_obj(&slabs[slab_idx]);
}

static void slab_free(void *addr, const char *file, int line)
{
    struct slab_ring *ring;

    if (addr == NULL || !is_obj_slab(addr)) {
        free(addr);
        return;
    }
    ring = get_slab_ring(addr);
    INC_SLAB_STAT(&ring->info->stats.frees);
    return_to_slab(addr, ring);
}

static void *slab_realloc(void *addr, size_t num, const char *file, int line)
{
    void *new;
    struct slab_ring *ring;

    if (addr == NULL)
        return slab_malloc(num, NULL, 0);

    if (!is_obj_slab(addr))
        return realloc(addr, num);
    ring = get_slab_ring(addr);
    if (num > MAX_SLAB) {
        new = malloc(num);
        if (new != NULL)
            memcpy(new, addr, ring->info->obj_size);
        slab_free(addr, NULL, 0);
        return new;
    }
    if (num <= ring->info->obj_size)
        return addr;
    new = slab_malloc(num, NULL, 0);
    if (new != NULL)
        memcpy(new, addr, ring->info->obj_size);
    slab_free(addr, NULL, 0);
    return new;
}

static void compute_slab_template(struct slab_info *slab)
{
    uint32_t bitmap_words  = 1; /* need at least one bitmap word */
    uint32_t obj_count;
    size_t objs_size;
    size_t available_size = (page_size - sizeof(struct slab_ring)) - (bitmap_words * sizeof(uint64_t));
    int word_size_increased;
    int i;

    for (obj_count = 1; ; obj_count++) {
        word_size_increased = 0;
        if ((obj_count % 64) == 0) {
            bitmap_words++;
            available_size = (page_size - sizeof(struct slab_ring)) - (bitmap_words * sizeof(uint64_t));
            word_size_increased = 1;
        }
        objs_size = obj_count * slab->obj_size;
        if (objs_size > available_size) {
            /* we went too far */
            if (word_size_increased == 1)
                bitmap_words--;
            obj_count--;
            break;
        }
    }
    slab->template.available_objs = obj_count;
    slab->template.bitmap_word_count = bitmap_words;
    slab->template.last_word_mask = (uint64_t)1 << (slab->template.available_objs % 64);
    for (i = slab->template.available_objs % 64; i < 64; i++)
        slab->template.last_word_mask = slab->template.last_word_mask | (uint64_t)1 << i;
    return;
}

static __attribute__((constructor)) void setup_slab_allocator()
{
    int i;

    fprintf(stderr, "Setting up slab allocator\n");
    if (!CRYPTO_set_mem_functions(slab_malloc, slab_realloc, slab_free))
        fprintf(stderr, "Failed to setup slag allocator\n");
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1)
        fprintf(stderr, "Failed to get page size\n");
    for (i = 0; i <= MAX_SLAB_IDX; i++) {
        compute_slab_template(&slabs[i]);
    }
}

static __attribute__((destructor)) void slab_cleanup()
{
#ifdef SLAB_STATS
    FILE *fp = stderr;
    char *path = getenv("SLAB_ALLOCATOR_LOG");
    int i;

    if (path != NULL) {
        fp = fopen(path, "w");
        if (fp == NULL)
            fp = stderr;
    }
    fprintf(fp, "{\"slabs\": [");

    for (i = 0; i <= MAX_SLAB_IDX; i++) {
        fprintf(fp, "{\"obj_size\":%lu, \"objs_per_slab\":%lu, \"allocs\":%lu, \"frees\":%lu, \"slabs\":%lu}",
            slabs[i].obj_size, slabs[i].template.available_objs, slabs[i].stats.allocs, slabs[i].stats.frees, slabs[i].stats.slabs);
        if (i != MAX_SLAB_IDX)
            fprintf(fp,",");
    }
    fprintf(fp, "]}");
    if (fp != stderr)
        fclose(fp);
#endif
}

