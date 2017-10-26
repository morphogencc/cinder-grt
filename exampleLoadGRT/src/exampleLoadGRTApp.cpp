#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "GRT.h"

using namespace ci;
using namespace ci::app;
using namespace std;
using namespace GRT;

class exampleLoadGRTApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
};

void exampleLoadGRTApp::setup() {
	//Generate a basic dummy dataset with 1000 samples, 5 classes, and 3 dimensions
	cout << "Generating dataset..." << endl;
	ClassificationData::generateGaussDataset("data.csv", 1000, 5, 3);

	//Load some training data from a file
	ClassificationData trainingData;

	cout << "Loading dataset..." << endl;
	if (!trainingData.load("data.csv")) {
		cout << "ERROR: Failed to load training data from file\n";
		return;
	}

	cout << "Data Loaded" << endl;

	//Print out some stats about the training data
	trainingData.printStats();

	//Partition the training data into a training dataset and a test dataset. 80 means that 80%
	//of the data will be used for the training data and 20% will be returned as the test dataset
	cout << "Splitting data into training/test split..." << endl;
	ClassificationData testData = trainingData.split(80);

	//Create a new Gesture Recognition Pipeline using an Adaptive Naive Bayes Classifier
	GestureRecognitionPipeline pipeline;
	pipeline.setClassifier(ANBC());

	//Train the pipeline using the training data
	cout << "Training model..." << endl;
	if (!pipeline.train(trainingData)) {
		cout << "ERROR: Failed to train the pipeline!\n";
		return;
	}

	//Save the pipeline to a file
	if (!pipeline.save("HelloWorldPipeline")) {
		cout << "ERROR: Failed to save the pipeline!\n";
		return;
	}

	//Load the pipeline from a file
	if (!pipeline.load("HelloWorldPipeline")) {
		cout << "ERROR: Failed to load the pipeline!\n";
		return;;
	}

	//Test the pipeline using the test data
	cout << "Testing model..." << endl;
	if (!pipeline.test(testData)) {
		cout << "ERROR: Failed to test the pipeline!\n";
		return;;
	}

	//Print some stats about the testing
	cout << "Test Accuracy: " << pipeline.getTestAccuracy() << endl;

	cout << "Precision: ";
	for (UINT k = 0; k<pipeline.getNumClassesInModel(); k++) {
		UINT classLabel = pipeline.getClassLabels()[k];
		cout << "\t" << pipeline.getTestPrecision(classLabel);
	}cout << endl;

	cout << "Recall: ";
	for (UINT k = 0; k<pipeline.getNumClassesInModel(); k++) {
		UINT classLabel = pipeline.getClassLabels()[k];
		cout << "\t" << pipeline.getTestRecall(classLabel);
	}cout << endl;

	cout << "FMeasure: ";
	for (UINT k = 0; k<pipeline.getNumClassesInModel(); k++) {
		UINT classLabel = pipeline.getClassLabels()[k];
		cout << "\t" << pipeline.getTestFMeasure(classLabel);
	}cout << endl;

	MatrixFloat confusionMatrix = pipeline.getTestConfusionMatrix();
	cout << "ConfusionMatrix: \n";
	for (UINT i = 0; i<confusionMatrix.getNumRows(); i++) {
		for (UINT j = 0; j<confusionMatrix.getNumCols(); j++) {
			cout << confusionMatrix[i][j] << "\t";
		}cout << endl;
	}
}

void exampleLoadGRTApp::mouseDown( MouseEvent event )
{
}

void exampleLoadGRTApp::update() {
}

void exampleLoadGRTApp::draw() {
	gl::clear( Color( 0, 0, 0 ) ); 
}

CINDER_APP(exampleLoadGRTApp, RendererGl, [](App::Settings* settings) {
	settings->setConsoleWindowEnabled();
});