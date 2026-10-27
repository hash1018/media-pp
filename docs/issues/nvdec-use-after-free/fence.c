// fence.so: an LD_PRELOAD that makes a read of a freed large block fault
// every time. Blocks of FENCE_MIN bytes or more get mappings of their own;
// freeing one makes it PROT_NONE for QUARANTINE_S seconds before it is
// unmapped, so its address is not handed out again meanwhile. Each fenced
// block's size, allocating and freeing callers and times are kept in
// `fence_log`, a static array a core's reader can print.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define FENCE_MIN (1000 * 1000)
#define QUARANTINE_S 30
#define MAGIC 0xfe11cefe11cefe11ULL
#define PAGE 4096
#define LOG 8192

extern void *__libc_malloc(size_t);
extern void __libc_free(void *);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);

struct header {
    uint64_t magic;
    uint64_t size;  // what was asked for
};

struct entry {
    void *base;
    size_t len;
    size_t size;
    void *user;
    void *alloc_caller;
    void *free_caller;
    double alloc_at, free_at;
    int freed, unmapped;
};

struct entry fence_log[LOG];
unsigned long fence_count;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static struct header *header_of(void *p) {
    return (struct header *)((char *)p - sizeof(struct header));
}

static int ours(void *p) {
    // A pointer of ours sits on a page boundary with the header just before
    // it, on the same mapping; glibc's never sits on one (its mmapped chunks
    // are offset by 16 bytes, its heap chunks by at least that).
    if (!p || ((uintptr_t)p & (PAGE - 1))) return 0;
    return header_of(p)->magic == MAGIC;
}

// Unmaps quarantined blocks older than QUARANTINE_S. Under `lock`.
static void purge(double t) {
    for (unsigned long i = 0; i < LOG; i++) {
        struct entry *e = &fence_log[i];
        if (e->freed && !e->unmapped && t - e->free_at > QUARANTINE_S) {
            munmap(e->base, e->len);
            e->unmapped = 1;
        }
    }
}

static void *fenced(size_t size, void *caller) {
    size_t len = ((size + PAGE - 1) / PAGE + 1) * PAGE;
    char *base = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        errno = ENOMEM;
        return NULL;
    }
    void *user = base + PAGE;
    header_of(user)->magic = MAGIC;
    header_of(user)->size = size;
    pthread_mutex_lock(&lock);
    double t = now();
    purge(t);
    // A slot not in use, or else the oldest unmapped one.
    struct entry *e = &fence_log[fence_count % LOG];
    for (unsigned long k = 0; k < LOG && e->base && !e->unmapped; k++)
        e = &fence_log[(fence_count + k) % LOG];
    fence_count++;
    *e = (struct entry){base, len, size, user, caller, NULL, t, 0, 0, 0};
    pthread_mutex_unlock(&lock);
    return user;
}

static void unfence(void *user, void *caller) {
    pthread_mutex_lock(&lock);
    double t = now();
    for (unsigned long i = 0; i < LOG; i++) {
        struct entry *e = &fence_log[i];
        if (e->user == user && !e->freed) {
            e->freed = 1;
            e->free_at = t;
            e->free_caller = caller;
            mprotect(e->base, e->len, PROT_NONE);
            break;
        }
    }
    purge(t);
    pthread_mutex_unlock(&lock);
}

size_t malloc_usable_size(void *p) {
    static size_t (*real)(void *);
    if (ours(p)) return header_of(p)->size;
    if (!real) real = (size_t (*)(void *))dlsym(RTLD_NEXT, "malloc_usable_size");
    return real(p);
}

void *malloc(size_t size) {
    if (size >= FENCE_MIN) return fenced(size, __builtin_return_address(0));
    return __libc_malloc(size);
}

void *calloc(size_t n, size_t size) {
    size_t total;
    if (__builtin_mul_overflow(n, size, &total)) {
        errno = ENOMEM;
        return NULL;
    }
    if (total >= FENCE_MIN) return fenced(total, __builtin_return_address(0));
    return __libc_calloc(n, size);
}

void free(void *p) {
    if (ours(p)) unfence(p, __builtin_return_address(0));
    else __libc_free(p);
}

void *realloc(void *p, size_t size) {
    if (!p) return malloc(size);
    if (ours(p) || size >= FENCE_MIN) {
        size_t old = ours(p) ? header_of(p)->size : malloc_usable_size(p);
        void *n = size >= FENCE_MIN ? fenced(size, __builtin_return_address(0)) : __libc_malloc(size);
        if (!n) return NULL;
        memcpy(n, p, old < size ? old : size);
        free(p);
        return n;
    }
    return __libc_realloc(p, size);
}

// Large aligned requests up to a page's alignment are fenced too: a page
// boundary is aligned to anything up to it.
void *memalign(size_t align, size_t size) {
    if (size >= FENCE_MIN && align <= PAGE) return fenced(size, __builtin_return_address(0));
    return __libc_memalign(align, size);
}

int posix_memalign(void **out, size_t align, size_t size) {
    void *p = memalign(align, size);
    if (!p) return ENOMEM;
    *out = p;
    return 0;
}

void *aligned_alloc(size_t align, size_t size) {
    return memalign(align, size);
}

void *valloc(size_t size) {
    return memalign(PAGE, size);
}

void *pvalloc(size_t size) {
    return memalign(PAGE, (size + PAGE - 1) & ~(size_t)(PAGE - 1));
}
