#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
using namespace std;
#include <signal.h>
#ifdef _MSC_VER
#include "VisualStudio/getopt/getopt.h"
#else
#include <getopt.h>
#endif

#ifndef _MSC_VER
#define _strdup(x) strdup(x)
#endif

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "network.h"
#include "gpu-cuda-miner.h"

bool longpoll = false;
char *address;
uint64_t *blockHeadermobj = nullptr;
uint64_t *headerHashmobj = nullptr;
uint64_t *nonceOutmobj = nullptr;
uint64_t *vpre = nullptr;
cudaError_t ret;
cudaStream_t cudastream;

unsigned int blocks_mined = 0;
static volatile int quit = 0;
bool target_corrupt_flag = false;

#if defined _MSC_VER
#define rotr64(x, n) _rotr64(x, n)
#else
#define rotr64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#endif

void quitSignal(int __unused)
{
	quit = 1;
	printf("\nquitting...\n");
}

static void printhexbytes(uint8_t *data, size_t sizeofdata)
{
	char *buffer = new char[2 * sizeofdata + 1];
	buffer[2 * sizeofdata] = '\0';

	for(int j = 0; j < sizeofdata; j++)
		sprintf(&buffer[2 * j], "%02x", data[j]);
	printf("%s", buffer);
	delete[] buffer;
}

// Perform global_item_size * iter_per_thread hashes
// Return -1 if a block is found
// Else return the hashrate in MH/s
static double grindNonces(uint64_t items_per_iter, int cycles_per_iter)
{
	static bool init = false;
	static uint64_t *headerHash = nullptr;
	static uint32_t *target = nullptr;
	static uint64_t *nonceOut = nullptr;
	static uint8_t *blockHeader = nullptr;
	static uint64_t *v1 = nullptr;

	if(!init)
	{
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		cudaMallocHost(&headerHash, 32 * MAXRESULTS);
		cudaMallocHost(&target, 32);
		cudaMallocHost(&nonceOut, 8 * MAXRESULTS);
		cudaMallocHost(&blockHeader, 80);
		cudaMallocHost(&v1, 16 * 8);
		ret = cudaGetLastError();
		if(ret != cudaSuccess)
		{
			printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
		}
		init = true;
	}

	// Start timing this iteration
	chrono::time_point<chrono::system_clock> startTime, endTime;
	startTime = chrono::system_clock::now();

	int i;

	// Get new block header and target
	if(get_header_for_work((uint8_t*)target, blockHeader) != 0)
	{
		return 0;
	}

	// Check for target corruption
	if(!longpoll && target[0] != 0)
	{
		if(target_corrupt_flag)
		{
			return -1;
		}
		target_corrupt_flag = true;
		printf("\nReceived corrupt target from Sia\n");
		printf("Usually this resolves itself within a minute or so\n");
		printf("If it happens frequently trying increasing seconds per iteration\n");
		printf("e.g. \"./gpu-miner -s 3 -c 200\"\n");
		printf("Waiting for problem to be resolved...");
		fflush(stdout);
		return -1;
	}
	target_corrupt_flag = 0;

	v1[0] = 0xBB1838E7A0A44BF9u + ((uint64_t*)blockHeader)[0]; v1[12] = rotr64(0x510E527FADE68281u ^ v1[0], 32); v1[8] = 0x6a09e667f3bcc908u + v1[12]; v1[4] = rotr64(0x510e527fade682d1u ^ v1[8], 24);
	v1[0] = v1[0] + v1[4] + ((uint64_t*)blockHeader)[1];       v1[12] = rotr64(v1[12] ^ v1[0], 16);              v1[8] = v1[8] + v1[12];               v1[4] = rotr64(v1[4] ^ v1[8], 63);
	v1[1] = 0x566D1711B009135Au + ((uint64_t*)blockHeader)[2]; v1[13] = rotr64(0x9b05688c2b3e6c1fu ^ v1[1], 32); v1[9] = 0xbb67ae8584caa73bu + v1[13]; v1[5] = rotr64(0x9b05688c2b3e6c1fu ^ v1[9], 24);
	v1[1] = v1[1] + v1[5] + ((uint64_t*)blockHeader)[3];       v1[13] = rotr64(v1[13] ^ v1[1], 16);              v1[9] = v1[9] + v1[13];               v1[5] = rotr64(v1[5] ^ v1[9], 63);

	ret = cudaMemcpyAsync(vpre, v1, 16 * 8, cudaMemcpyHostToDevice, cudastream);
	if(ret != cudaSuccess)
	{
		printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
	}

	for(i = 0; i < cycles_per_iter; i++)
	{
		// Copy input data to the memory buffer
		ret = cudaMemcpyAsync(blockHeadermobj, blockHeader, 80, cudaMemcpyHostToDevice, cudastream);
		if(ret != cudaSuccess)
		{
			printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
		}
		ret = cudaMemsetAsync(nonceOutmobj, 0, 8 * MAXRESULTS, cudastream);
		if(ret != cudaSuccess)
		{
			printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
		}

		nonceGrindcuda(cudastream, items_per_iter, blockHeadermobj, headerHashmobj, nonceOutmobj, vpre, swap64(*(uint64_t*)target));
		// Copy result to host
		ret = cudaMemcpyAsync(headerHash, headerHashmobj, 32 * MAXRESULTS, cudaMemcpyDeviceToHost, cudastream);
		if(ret != cudaSuccess)
		{
			printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
		}

		ret = cudaMemcpyAsync(nonceOut, nonceOutmobj, 8 * MAXRESULTS, cudaMemcpyDeviceToHost, cudastream);
		if(ret != cudaSuccess)
		{
			printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
		}
		ret = cudaStreamSynchronize(cudastream);
		if(ret != cudaSuccess)
		{
			printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
		}
		bool found = false;
		int k = 0;
		while(k<MAXRESULTS && nonceOut[k] != 0)
		{
			int j = 0;
			while(j < 4 && swap64(headerHash[k * 4 + j]) == ((uint64_t*)target)[j])
				j++;
			if(j == 4 || swap64(headerHash[k * 4 + j]) < ((uint64_t*)target)[j])
			{
				/* debug
				printf("\n");
				printhexbytes((uint8_t*)target, 8);
				printf(" target\n");
				printhexbytes((uint8_t*)(headerHash + k * 4), 8);
				printf(" hash\n");
				*/
				// Copy nonce to header.
				((uint64_t*)blockHeader)[4] = nonceOut[k];
				if(submit_header(blockHeader))
					blocks_mined++;
				found = true;
			}
			k++;
		}
		if(found)
			return -1;
		if(quit)
		{
			return 0;
		}
	}

	// Hashrate is inaccurate if a block was found
	endTime = chrono::system_clock::now();
	double elapsedTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count() / 1000000.0;
	double hash_rate = cycles_per_iter * (double)items_per_iter / elapsedTime / 1000000;

	return hash_rate;
}

static int msver(void)
{
#ifdef _MSC_VER
	switch(_MSC_VER)
	{
		case 1500: return 2008;
		case 1600: return 2010;
		case 1700: return 2012;
		case 1800: return 2013;
		case 1900: return 2015;
		default: return (_MSC_VER / 100);
	}
#else
	return 0;
#endif
}

int main(int argc, char *argv[])
{
	int c;
	char *tmp;
	unsigned int deviceid = 0;
	cudaDeviceProp deviceProp;
	char *serverip = (char *)"localhost";
	char *port_number = (char *)"9980";
	char *useragent = (char *)"Sia-Agent";
	address = nullptr;
	double hash_rate;
	uint64_t items_per_iter = 256 * 256 * 256 * 16;

	// parse args
	unsigned int cycles_per_iter = 15;
	double seconds_per_iter = 10.0;

#if defined _WIN64 || defined _LP64
	printf("\nSia-CUDA-Miner 4.01 (64bit)\n");
#else
	printf("\nSia-CUDA-Miner 4.01 (32bit)\n");
#endif
#ifdef _MSC_VER
	printf("Compiled with Visual C++ %d\n", msver());
#else
#ifdef __clang__
	printf("Compiled with Clang %s\n", __clang_version__);
#else
#ifdef __GNUC__
	printf("Compiled with GCC %d.%d\n", __GNUC__, __GNUC_MINOR__);
#else
	printf("Compiled with an unusual compiler\n");
#endif
#endif
#endif
	printf("Using Nvidia CUDA Toolkit %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

	while((c = getopt(argc, argv, "hc:s:p:d:u:lq:")) != -1)
	{
		switch(c)
		{
			case 'h':
				printf("\nUsage:\n\n");
				printf("  -l           : enable pool mining (longpolling)\n");
				printf("  -q <address> : wallet address used for pool mining\n");
				printf("\n");
				printf("  -u <url>     : pool url\n");
				printf("\n");
				printf("  -p <port>    : port number\n");
				printf("                 default: %s\n", port_number);
				printf("\n");
				printf("  -c <cycles>  : number of hashing loops between API calls\n");
				printf("                 default: %d\n", cycles_per_iter);
				printf("                 Increase this if your computer is freezing or locking up\n");
				printf("\n");
				printf("  -s <seconds> : seconds between Sia API calls and hash rate updates\n");
				printf("                 default: %f\n", seconds_per_iter);
				printf("\n");
				printf("  -d <device>  : the device id of the card you want to use\n");
				printf("                 default: 0\n");
				printf("\n");
				exit(0);
				break;
			case 'l':
				longpoll = true;;
				break;
			case 'c':
				cycles_per_iter = strtoul(optarg, &tmp, 10);
				if(cycles_per_iter < 1 || cycles_per_iter > 1000)
				{
					printf("Cycles must be at least 1 and no more than 1000\n");
					exit(1);
				}
				break;
			case 's':
				seconds_per_iter = strtod(optarg, &tmp);
				break;
			case 'u':
				serverip = _strdup(optarg);
				break;
			case 'p':
				port_number = _strdup(optarg);
				break;
			case 'd':
				deviceid = strtoul(optarg, &tmp, 10);
				break;
			case 'q':
				address = _strdup(optarg);
				break;
		}
	}
	if(longpoll && address == nullptr)
	{
		printf("\nWallet address needed for pool mining !\n");
		exit(1);
	}

	// Set siad URL
	network_init(serverip, port_number, useragent);

	printf("\nInitializing...\n");

	int version;
	ret = cudaDriverGetVersion(&version);
	if(ret != cudaSuccess)
	{
		printf("Unable to query CUDA driver version! Is an nVidia driver installed?\n");
		exit(1);
	}

	if(version < CUDART_VERSION)
	{
		printf("Driver does not support CUDA %d.%d API! Update your nVidia driver!\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
		exit(1);
	}

	int deviceCount;
	ret = cudaGetDeviceCount(&deviceCount);
	if(ret != cudaSuccess)
	{
		if(ret == cudaErrorNoDevice)
			printf("No CUDA device found");
		if(ret == cudaErrorInsufficientDriver)
			printf("Driver error\n");
		return -1;
	}
	for(int device = 0; device<deviceCount; ++device)
	{
		ret = cudaGetDeviceProperties(&deviceProp, device);
		if(ret != cudaSuccess)
		{
			printf("CUDA error in %s line %d: %s\n", __FILE__ , __LINE__,cudaGetErrorString(ret)); exit(1);
		}
		printf("Device %d: %s (Compute Capability %d.%d)\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
	}
	printf("\nUsing device %d\n", deviceid);

	ret = cudaSetDevice(deviceid);
	if(ret != cudaSuccess)
	{
		printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
	}
	cudaDeviceReset();
	ret = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	if(ret != cudaSuccess)
	{
		printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
	}

	ret = cudaStreamCreate(&cudastream);
	if(ret != cudaSuccess)
	{
		printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
	}
	// Create Buffer Objects
	ret = cudaMalloc(&blockHeadermobj, 80);
	if(ret != cudaSuccess)
	{
		printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
	}
	ret = cudaMalloc(&headerHashmobj, 32 * MAXRESULTS);
	if(ret != cudaSuccess)
	{
		printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
	}
	ret = cudaMalloc(&nonceOutmobj, 8 * MAXRESULTS);
	if(ret != cudaSuccess)
	{
		printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
	}
	ret = cudaMalloc(&vpre, 16 * 8);
	if(ret != cudaSuccess)
	{
		printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
	}

	chrono::time_point<chrono::system_clock> startTime, endTime;
	startTime = chrono::system_clock::now();

	grindNonces(items_per_iter, 1);

	endTime = chrono::system_clock::now();
	double elapsedTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count() / 1000000.0;
	items_per_iter *= (seconds_per_iter / elapsedTime) / cycles_per_iter;

	// Grind nonces until SIGINT
	signal(SIGINT, quitSignal);
	while(!quit)
	{
		// Repeat until no block is found
		do
		{
			hash_rate = grindNonces(items_per_iter, cycles_per_iter);
		} while(hash_rate == -1);

		if(!quit && hash_rate != 0)
		{
			if(!longpoll)
				printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
			else
				printf("\rMining at %.3f MH/s\t%u shares mined", hash_rate, blocks_mined);
			fflush(stdout);
		}
	}

	// Finalization
	ret = cudaStreamDestroy(cudastream);
	if(ret != cudaSuccess)
	{
		printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); exit(1);
	}
	cudaProfilerStop();
	cudaDeviceReset();

	network_cleanup();

	return EXIT_SUCCESS;
}
