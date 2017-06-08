#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "nvml.h"
#include "ofMain.h"

//https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html

#define DEBUG

#ifndef NO_NVML
#include "nvml.h"
#endif // ndef NO_NVML

#define CUDA_CALL(function, ...)  { \
    cudaError_t status = function(__VA_ARGS__); \
    anyCheck(status == cudaSuccess, cudaGetErrorString(status), #function, __FILE__, __LINE__); \
}

#ifndef NO_NVML
#define NVML_CALL(function, ...)  { \
    nvmlReturn_t status = function(__VA_ARGS__); \
    anyCheck(status == NVML_SUCCESS, nvmlErrorString(status), #function, __FILE__, __LINE__); \
}
#else // ndef NO_NVML
#define NVML_CALL(function, ...) { } // We create dummy wrapper to skip initialization code etc.
#endif // ndef NO_NVML

struct NvidiaCuda
{
	string ID;
	string nvmlID;
	string pciDomainID;
	string pciBusID;
	string pciDeviceID;
	string devName;
	string devMajor;
	string devMinor;
	string memUsed;
	string memTot;
	string proCount;
	string clockRate;
};

class CUDAMonitor : public ofThread
{
	public:
	int cudaDeviceCount;
	struct cudaDeviceProp deviceProp;
	size_t memUsed, memTotal;

	#ifndef NO_NVML
	    unsigned int nvmlDeviceCount;
	    nvmlPciInfo_t nvmlPciInfo;
	    nvmlDevice_t nvmlDevice;
	#endif

	vector<NvidiaCuda> nvidia_data;

	vector<NvidiaCuda> getData()
	{
		return nvidia_data;
	}

	void setup()
	{
		startThread(true);
	}

	void draw(int x, int  y, int w=280, int h=200)
	{
		ofPushStyle();
		ofPushMatrix();
		for(int i = 0; i < nvidia_data.size(); i++)
		{
			ofFill();
			ofSetColor(ofColor::silver);
			ofDrawRectRounded(ofRectangle(x,y+(i*h),w,h),10);
			ofNoFill();
			ofSetColor(ofColor::black);
			ofDrawRectRounded(ofRectangle(x,y+(i*h),w,h),10);

			ofDrawBitmapStringHighlight("Device: "+nvidia_data[i].ID,x+15,y+25,ofColor(255,255,255,200),ofColor::black);
			ofDrawBitmapStringHighlight("SMI Device: "+nvidia_data[i].nvmlID,x+15,y+45,ofColor(255,255,255,200),ofColor::black);
			ofDrawBitmapStringHighlight("PCI: "+nvidia_data[i].pciDomainID+":"+nvidia_data[i].pciBusID+":"+nvidia_data[i].pciDeviceID,
						    x+15,y+65,ofColor(255,255,255,200),ofColor::black);
			ofDrawBitmapStringHighlight("Model Name: "+nvidia_data[i].devName,x+15,y+85,ofColor(255,255,255,200),ofColor::black);
			ofDrawBitmapStringHighlight("Version: "+nvidia_data[i].devMajor+"."+nvidia_data[i].devMinor,x+15,y+105,ofColor(255,255,255,200),ofColor::black);
			ofDrawBitmapStringHighlight("Mem Used: "+nvidia_data[i].memUsed+" MB",x+15,y+125,ofColor(255,255,255,200),ofColor::black);
			ofDrawBitmapStringHighlight("Mem Total: "+nvidia_data[i].memTot+" MB",x+15,y+145,ofColor(255,255,255,200),ofColor::black);
			ofDrawBitmapStringHighlight("Process Count: "+nvidia_data[i].proCount,x+15,y+165,ofColor(255,255,255,200),ofColor::black);
			ofDrawBitmapStringHighlight("Clock Rate: "+nvidia_data[i].clockRate,x+15,y+185,ofColor(255,255,255,200),ofColor::black);
		}
		ofPopMatrix();
		ofPopStyle();
	}

    	void threadedFunction() 
	{
	        while(isThreadRunning()) 
		{			
			CUDA_CALL(cudaGetDeviceCount, &cudaDeviceCount);
        	        NVML_CALL(nvmlInit);
	                NVML_CALL(nvmlDeviceGetCount, &nvmlDeviceCount);

			nvidia_data.clear();
			for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
				NvidiaCuda nvc;

			        CUDA_CALL(cudaGetDeviceProperties, &deviceProp, deviceId);
				nvc.ID = to_string(deviceId);
				#ifndef NO_NVML
				        int nvmlDeviceId = -1;
				        for (int nvmlId = 0; nvmlId < nvmlDeviceCount; ++nvmlId) {
				            NVML_CALL(nvmlDeviceGetHandleByIndex, nvmlId, &nvmlDevice);
				            NVML_CALL(nvmlDeviceGetPciInfo, nvmlDevice, &nvmlPciInfo);
				            if (deviceProp.pciDomainID == nvmlPciInfo.domain &&
				                deviceProp.pciBusID    == nvmlPciInfo.bus    &&
				                deviceProp.pciDeviceID == nvmlPciInfo.device) {
					                nvmlDeviceId = nvmlId;
					                break;
				            }
        				}
					nvc.nvmlID = to_string(nvmlDeviceId);
				#endif
				nvc.pciDomainID = to_string(deviceProp.pciDomainID);
				nvc.pciBusID    = to_string(deviceProp.pciBusID);
				nvc.pciDeviceID = to_string(deviceProp.pciDeviceID);
				nvc.devName  = deviceProp.name;
				nvc.devMajor = to_string(deviceProp.major);
				nvc.devMinor = to_string(deviceProp.minor);
				#ifndef NO_NVML
			        getMemoryUsageNVML(nvmlDevice, memUsed, memTotal);
				#else 
				getMemoryUsageCUDA(deviceId, memUsed, memTotal);
				#endif
				nvc.memUsed    = to_string(memUsed);
				nvc.memTot     = to_string(memTotal);
				nvc.proCount   = to_string(deviceProp.multiProcessorCount);
				nvc.clockRate  = to_string(deviceProp.clockRate);

				nvidia_data.push_back(nvc);
			}
	                NVML_CALL(nvmlShutdown);
			ofSleepMillis(1000);
		}
	}

	void exit()
	{
		stopThread();
	}	

	private:
	void anyCheck(bool is_ok, const char *description, const char *function, const char *file, int line) {
	    if (!is_ok) {
		fprintf(stderr,"Error: %s in %s at %s:%d\n", description, function, file, line);
	    }
	}

	void getMemoryUsageCUDA(int deviceId, size_t &memUsed, size_t &memTotal) {
	    size_t memFree;
	    CUDA_CALL(cudaSetDevice, deviceId);
	    CUDA_CALL(cudaMemGetInfo, &memFree, &memTotal);
	    memUsed = memTotal - memFree;
	    memUsed = memUsed / 1024 / 1024;
	    memTotal = memTotal / 1024 / 1024;
	}

	#ifndef NO_NVML
	void getMemoryUsageNVML(nvmlDevice_t &nvmlDevice, size_t &memUsed, size_t &memTotal) {
	    nvmlMemory_t nvmlMemory;
	    NVML_CALL(nvmlDeviceGetMemoryInfo, nvmlDevice, &nvmlMemory);
	    memUsed = nvmlMemory.used / 1024 / 1024;
	    memTotal = nvmlMemory.total / 1024 / 1024;
	}
	#endif

};
/* end */
