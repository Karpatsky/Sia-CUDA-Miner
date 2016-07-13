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

void set_port(char *port);
int get_header_for_work(uint8_t *target, uint8_t *header);
bool submit_header(uint8_t *header);
void network_init(const char *domain, const char *port, const char *useragent);
void network_cleanup(void);
