#include "ofMain.h"
#include "smi.h"

class ofApp : public ofBaseApp
{
	public:
	CUDAMonitor cmon;

	void setup()
	{
		cmon.setup();
	}
	
	void draw()
	{
		ofBackground(ofColor::black);
		cmon.draw(10,10);
	}

	void exit()
	{
		cmon.exit();
	}
};

int main(int argc, char *argv[])
{
        ofSetupOpenGL(1024, 768, OF_WINDOW);
        ofRunApp( new ofApp());
}
