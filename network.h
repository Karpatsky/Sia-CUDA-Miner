#include <cstdint>
#include <curl/curl.h>

#ifndef WIN32
#include <unistd.h>
#define Sleep(duration) usleep((duration)*1000)
#endif

struct inData
{
	uint8_t *bytes;
	size_t len;
};

#if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#define LINUX_BSWAP
#endif

static inline uint32_t swap32(uint32_t x)
{
#ifdef LINUX_BSWAP
	return __builtin_bswap32(x);
#else
#ifdef _MSC_VER
	return _byteswap_ulong(x);
#else
	return ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu));
#endif
#endif
}

#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint64_t swap64(uint64_t x)
{
	uint64_t result;
	uint2 t;
	asm("mov.b64 {%0,%1},%2; \n\t"
			: "=r"(t.x), "=r"(t.y) : "l"(x));
	t.x = __byte_perm(t.x, 0, 0x0123);
	t.y = __byte_perm(t.y, 0, 0x0123);
	asm("mov.b64 %0,{%1,%2}; \n\t"
			: "=l"(result) : "r"(t.y), "r"(t.x));
	return result;
}
#else
/* host */
#ifdef __GNUC__
#if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#define swap64(x) __builtin_bswap64(x)
#endif
#else
#ifdef _MSC_VER
#define swap64(x) _byteswap_uint64(x)
#else
#define swap64(x) \
				((uint64_t)((((uint64_t)(x)) >> 56) | \
				(((uint64_t)(x) & 0x00ff000000000000ULL) >> 40) | \
				(((uint64_t)(x) & 0x0000ff0000000000ULL) >> 24) | \
				(((uint64_t)(x) & 0x000000ff00000000ULL) >>  8) | \
				(((uint64_t)(x) & 0x00000000ff000000ULL) <<  8) | \
				(((uint64_t)(x) & 0x0000000000ff0000ULL) << 24) | \
				(((uint64_t)(x) & 0x000000000000ff00ULL) << 40) | \
				(((uint64_t)(x)) << 56)))
#endif
#endif
#endif

void set_port(char *port);
int get_header_for_work(uint8_t *target, uint8_t *header);
bool submit_header(uint8_t *header);
void network_init(const char *domain, const char *port, const char *useragent);
void network_cleanup(void);
