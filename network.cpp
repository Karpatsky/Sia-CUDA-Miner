#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
using namespace std;
#include "network.h"

extern bool longpoll;
extern char *address;

static char bfw_url[255], submit_url[255];
static CURL *curl;
static char curlerrorbuffer[CURL_ERROR_SIZE];
struct inData in;

static double target_to_diff(const uint32_t *const target)
{
	//	return (4294967296.0 * 0xffff0000) / ((double)swap32(target[2]) + ((double)swap32(target[1]) * 4294967296.0));
	return pow(2.0, 8 * 32) / (((((((swap32(target[0])
		* 4294967296.0 + swap32(target[1]))
		* 4294967296.0 + swap32(target[2]))
		* 4294967296.0 + swap32(target[3]))
		* 4294967296.0 + swap32(target[4]))
		* 4294967296.0 + swap32(target[5]))
		* 4294967296.0 + swap32(target[6]))
		* 4294967296.0 + swap32(target[7]));
}

// Write network data to an array of bytes
size_t writefunc(void *ptr, size_t size, size_t nmemb, struct inData *in)
{
	size_t new_len = size*nmemb;
	if(in == NULL || new_len == 0)
		return 0;

	in->bytes = (uint8_t*)realloc(in->bytes, in->len + new_len);
	if(in->bytes == NULL)
	{
		fprintf(stderr, "malloc() failed\n");
		exit(EXIT_FAILURE);
	}
	memcpy(in->bytes + in->len, ptr, new_len);
	in->len += new_len;

	return new_len;
}

void network_init(const char *domain, const char *port, const char *useragent)
{
	CURLcode res;

	curl = curl_easy_init();
	if(curl == NULL)
	{
		printf("\nError: can't init curl\n");
		exit(EXIT_FAILURE);
	}
/*
	res = curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	if(res != CURLE_OK)
	{
		printf("verbose res=%d\n", res);
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}
*/
	res = curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curlerrorbuffer);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curl_easy_strerror(res));
		curl_easy_cleanup(curl);
		exit(1);
	}

	if(bfw_url == NULL || submit_url == NULL)
	{
		printf("\nmalloc error\n");
		exit(EXIT_FAILURE);
	}

	if(!longpoll)
	{
		sprintf(bfw_url, "http://%s:%s/miner/header", domain, port);
		sprintf(submit_url, "http://%s:%s/miner/header", domain, port);
	}
	else
	{
		sprintf(bfw_url, "http://%s:%s/miner/header?address=%s", domain, port, address);
		sprintf(submit_url, "http://%s:%s/miner/header?address=%s", domain, port, address);
	}

	res = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}

	res = curl_easy_setopt(curl, CURLOPT_WRITEDATA, &in);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}

	res = curl_easy_setopt(curl, CURLOPT_USERAGENT, useragent);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}
}

void network_cleanup(void)
{
	curl_easy_cleanup(curl);
}

static int check_http_response(CURL *curl)
{
	long http_code = 0;
	CURLcode err = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
	if(err == CURLE_OK)
	{
		if(http_code != 200 && http_code != 204 && http_code != 0)
		{
			fprintf(stderr, "\nHTTP error %lu", http_code);
			if(!longpoll && http_code == 400)
			{
				fprintf(stderr, "\nplease unlock the wallet");
				Sleep(10000);
			}
			if(in.len > 0)
			{
				in.bytes = (uint8_t*)realloc(in.bytes, in.len + 1);
				*(in.bytes + in.len) = 0;
				printf(": %s", in.bytes);
			}
			printf("\n");
			return 1;
		}
	}
	return 0;
}

int get_header_for_work(uint8_t *target, uint8_t *header)
{
	static double diff = 0.0;
	double tmp;

	CURLcode res;
	in.bytes = NULL;
	in.len = 0;

	// Get data from siad
	curl_easy_setopt(curl, CURLOPT_POST, 0);
	res = curl_easy_setopt(curl, CURLOPT_URL, bfw_url);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}

	res = curl_easy_perform(curl);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}
	if(check_http_response(curl))
	{
		return 1;
	}
	if(in.len != 112)
	{
		fprintf(stderr, "\ncurl did not receive correct bytes (got %Iu, expected 112)\n", in.len);
		free(in.bytes);
		return 1;
	}

	// Copy data to return
	memcpy(target, in.bytes, 32);
	memcpy(header, in.bytes + 32, 80);
	tmp = target_to_diff((uint32_t*)target);
	if(tmp != diff)
	{
		double div;
		char *e;
		if(tmp < 100e15)
		{
			div = 1.0e12;
			e = "T";
		}
		if(tmp < 100e12)
		{
			div = 1.0e9;
			e = "G";
		}
		if(tmp < 100e9)
		{
			div = 1.0e6;
			e = "M";
		}
		if(tmp < 100e6)
		{
			div = 1.0e3;
			e = "k";
		}
		if(longpoll)
			printf("\ndifficulty = %u %sH/share\n", lround(tmp / div), e);
		else
			printf("\ndifficulty = %u %sH/block\n", lround(tmp / div), e);
		diff = tmp;
	}

	free(in.bytes);
	return 0;
}

bool submit_header(uint8_t *header)
{
	CURLcode res;
	in.len = 0;
	in.bytes = NULL;
		
	res = curl_easy_setopt(curl, CURLOPT_URL, submit_url);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}
	curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_0);
	curl_easy_setopt(curl, CURLOPT_POST, 1);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 80);
	res = curl_easy_setopt(curl, CURLOPT_POSTFIELDS, header);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}

	res = curl_easy_perform(curl);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		fprintf(stderr, "failed to submit header\n");
		curl_easy_cleanup(curl);
		exit(1);
	}
	if(check_http_response(curl) != 0)
	{
		free(in.bytes);
		return false;
	}
	else
	{
		if(in.len > 0)
		{
			fprintf(stderr, (char*)in.bytes);
		}
		free(in.bytes);
		return true;
	}
}
