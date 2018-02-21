#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "GRT.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class exampleClassificationApp : public App {
public:
	enum ClassifierType {
		ADABOOST = 0,
		DECISION_TREE,
		KNN,
		GAUSSIAN_MIXTURE_MODEL,
		NAIVE_BAYES,
		MINDIST,
		RANDOM_FOREST_10,
		RANDOM_FOREST_100,
		RANDOM_FOREST_200,
		SOFTMAX,
		SVM_LINEAR,
		SVM_RBF,
		NUM_CLASSIFIERS
	};

	void setup() override;
	void mouseDown(MouseEvent event) override;
	void keyDown(KeyEvent event) override;
	void update() override;
	void draw() override;
	bool setClassifier(const int type);
	std::string classifierTypeToString(const int type);

	int mClassifierType;
	GRT::ClassificationData mTrainingData;
	GRT::GestureRecognitionPipeline mPipeline;
	std::vector<ci::Color> mClassColors;
	UINT mTrainingClassLabel;
	bool mBuildTexture;
	ci::SurfaceRef mClassifyingSurface;
};


bool exampleClassificationApp::setClassifier(const int type) {

	GRT::AdaBoost adaboost;
	GRT::DecisionTree dtree;
	GRT::KNN knn;
	GRT::GMM gmm;
	GRT::ANBC naiveBayes;
	GRT::MinDist minDist;
	GRT::RandomForests randomForest;
	GRT::Softmax softmax;
	GRT::SVM svm;
	bool enableNullRejection = false;

	mClassifierType = type;
	mPipeline.clear();

	switch (mClassifierType) {
	case ADABOOST:
		adaboost.enableNullRejection(enableNullRejection);
		adaboost.setNullRejectionCoeff(3);
		mPipeline << adaboost;
		break;
	case DECISION_TREE:
		dtree.enableNullRejection(enableNullRejection);
		dtree.setNullRejectionCoeff(3);
		dtree.setMaxDepth(10);
		dtree.setMinNumSamplesPerNode(3);
		dtree.setRemoveFeaturesAtEachSplit(false);
		mPipeline << dtree;
		break;
	case KNN:
		knn.enableNullRejection(enableNullRejection);
		knn.setNullRejectionCoeff(3);
		mPipeline << knn;
		break;
	case GAUSSIAN_MIXTURE_MODEL:
		gmm.enableNullRejection(enableNullRejection);
		gmm.setNullRejectionCoeff(3);
		mPipeline << gmm;
		break;
	case NAIVE_BAYES:
		naiveBayes.enableNullRejection(enableNullRejection);
		naiveBayes.setNullRejectionCoeff(10);
		naiveBayes.enableScaling(true);
		naiveBayes.enableNullRejection(true);
		mPipeline << naiveBayes;
		break;
	case MINDIST:
		minDist.enableNullRejection(enableNullRejection);
		minDist.setNullRejectionCoeff(3);
		mPipeline << GRT::MinDist(false, true);
		break;
	case RANDOM_FOREST_10:
		randomForest.enableNullRejection(enableNullRejection);
		randomForest.setNullRejectionCoeff(3);
		randomForest.setForestSize(10);
		randomForest.setNumRandomSplits(2);
		randomForest.setMaxDepth(10);
		randomForest.setMinNumSamplesPerNode(5);
		randomForest.setRemoveFeaturesAtEachSplit(false);
		mPipeline << randomForest;
		break;
	case RANDOM_FOREST_100:
		randomForest.enableNullRejection(enableNullRejection);
		randomForest.setNullRejectionCoeff(3);
		randomForest.setForestSize(100);
		randomForest.setNumRandomSplits(2);
		randomForest.setMaxDepth(10);
		randomForest.setMinNumSamplesPerNode(3);
		randomForest.setRemoveFeaturesAtEachSplit(false);
		mPipeline << randomForest;
		break;
	case RANDOM_FOREST_200:
		randomForest.enableNullRejection(enableNullRejection);
		randomForest.setNullRejectionCoeff(3);
		randomForest.setForestSize(200);
		randomForest.setNumRandomSplits(2);
		randomForest.setMaxDepth(10);
		randomForest.setMinNumSamplesPerNode(3);
		randomForest.setRemoveFeaturesAtEachSplit(false);
		mPipeline << randomForest;
		break;
	case SOFTMAX:
		softmax.enableNullRejection(enableNullRejection);
		softmax.setNullRejectionCoeff(3);
		mPipeline << softmax;
		break;
	case SVM_LINEAR:
		svm.enableNullRejection(enableNullRejection);
		svm.setNullRejectionCoeff(3);
		mPipeline << GRT::SVM(GRT::SVM::LINEAR_KERNEL);
		break;
	case SVM_RBF:
		svm.enableNullRejection(enableNullRejection);
		svm.setNullRejectionCoeff(3);
		mPipeline << GRT::SVM(GRT::SVM::RBF_KERNEL);
		break;
	default:
		return false;
		break;
	}

	return true;
}

std::string exampleClassificationApp::classifierTypeToString(const int type) {
	switch (type) {
	case ADABOOST:
		return "ADABOOST";
		break;
	case DECISION_TREE:
		return "DECISION_TREE";
		break;
	case KNN:
		return "KNN";
		break;
	case GAUSSIAN_MIXTURE_MODEL:
		return "GMM";
		break;
	case NAIVE_BAYES:
		return "NAIVE_BAYES";
		break;
	case MINDIST:
		return "MINDIST";
		break;
	case RANDOM_FOREST_10:
		return "RANDOM_FOREST_10";
		break;
	case RANDOM_FOREST_100:
		return "RANDOM_FOREST_100";
		break;
	case RANDOM_FOREST_200:
		return "RANDOM_FOREST_200";
		break;
	case SOFTMAX:
		return "SOFTMAX";
		break;
	case SVM_LINEAR:
		return "SVM_LINEAR";
		break;
	case SVM_RBF:
		return "SVM_RBF";
		break;
	}
	return "UNKOWN_CLASSIFIER";
}

void exampleClassificationApp::setup() {
	mTrainingClassLabel = 1;
	mTrainingData.setNumDimensions(2);
	setClassifier(NAIVE_BAYES);

	mBuildTexture = false;
	mClassifyingSurface = ci::Surface::create(ci::app::getWindowWidth(), ci::app::getWindowHeight(), true, ci::SurfaceChannelOrder::RGBA);
	mClassColors.resize(4);
	mClassColors[0] = ci::Color(255, 0, 0);
	mClassColors[1] = ci::Color(0, 255, 0);
	mClassColors[2] = ci::Color(0, 0, 255);
	mClassColors[3] = ci::Color(255, 255, 255);
}

void exampleClassificationApp::mouseDown(MouseEvent event) {
	//Grab the current mouse x and y position
	GRT::VectorFloat sample(2);
	sample[0] = event.getPos().x / double(ci::app::getWindowWidth());
	sample[1] = event.getPos().y / double(ci::app::getWindowHeight());

	mTrainingData.addSample(mTrainingClassLabel, sample);
}

void exampleClassificationApp::keyDown(KeyEvent event) {
	if (event.getChar() == '1') {
		mTrainingClassLabel = 1;
	}
	else if (event.getChar() == '2') {
		mTrainingClassLabel = 2;
	}
	else if (event.getChar() == '3') {
		mTrainingClassLabel = 3;
	}
	else if (event.getChar() == 't') {
		if (mPipeline.train(mTrainingData)) {
			std::printf("Pipeline Trained!\n");
			mBuildTexture = true;
		}
		else {
			std::printf("Pipeline could not be trained.\n");
		}
	}
	else if (event.getChar() == 'c') {
		mTrainingData.clear();
		mPipeline.clearModel();
		setClassifier(NAIVE_BAYES);
		mBuildTexture = false;
	}
}


void exampleClassificationApp::update() {
	//If the pipeline has been trained, then run the prediction
	if (mBuildTexture) {
		//std::printf("Building texture...\n");
		unsigned int classLabel = 0;
		GRT::VectorFloat featureVector(2);
		GRT::VectorFloat likelihoods;
		float maximumLikelihood;
		const UINT numClasses = mPipeline.getNumClasses();
		for(int col = 0; col < ci::app::getWindowWidth(); col++) {
			for (int row = 0; row < ci::app::getWindowHeight(); row++) {
				featureVector[0] = float(col) / double(ci::app::getWindowWidth());
				featureVector[1] = float(row) / double(ci::app::getWindowHeight());
				if (mPipeline.predict(featureVector)) {
					classLabel = mPipeline.getPredictedClassLabel();
					if ((row % 50 == 0) && (col % 50 == 0)) {
						std::printf("\nProcessing Pixel (%d, %d)... \n", col, row);
						std::printf("Class Label: %d\n", classLabel);
					}
					maximumLikelihood = mPipeline.getMaximumLikelihood();
					likelihoods = mPipeline.getClassLikelihoods();
					mClassifyingSurface->setPixel(ci::ivec2(col, row), ci::ColorA(mClassColors[(classLabel - 1) % mClassColors.size()], 1.0));
				}
				else {
					std::printf("failed to predict.\n");
				}
			}
		}
		std::printf("Texture built!\n");
		mBuildTexture = false;
	}
}

void exampleClassificationApp::draw() {
	gl::clear(Color(0, 0, 0));

	if (mPipeline.getTrained()) {
		// draw classification surface
		gl::Texture2dRef texture = gl::Texture2d::create(*mClassifyingSurface);
		gl::draw(texture, getWindowBounds());
	}

	for (unsigned int i = 0; i < mTrainingData.getNumSamples(); i++) {
		// draw previously recorded samples
		float x = mTrainingData[i][0] * ci::app::getWindowWidth();
		float y = mTrainingData[i][1] * ci::app::getWindowHeight();

		gl::color(0,0,0);
		ci::gl::drawSolidCircle(ci::vec2(x, y), 6);

		gl::color(mClassColors[mTrainingData[i].getClassLabel() - 1 % mClassColors.size()]);
		ci::gl::drawSolidCircle(ci::vec2(x, y), 5);
	}
}

CINDER_APP(exampleClassificationApp, RendererGl, [](App::Settings* settings) {
	settings->setWindowSize(480,320);
	settings->setConsoleWindowEnabled(); 
})