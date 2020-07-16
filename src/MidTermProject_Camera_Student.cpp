/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results
    
    bool writeFile = false;
    std::vector<string> detectors_names;
    std::vector<string> descriptors_name;
    
    if(writeFile) {
        std::vector<string> detectors{"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT", "SURF"};
        std::vector<string> descriptors{"BRISK","SIFT","BRIEF", "ORB", "FREAK", "AKAZE"};
        detectors_names =  detectors;
        descriptors_name = descriptors;
    }
    else {
        std::vector<string> detector{"SHITOMASI"};
        std::vector<string> descriptor{"BRISK"};
        detectors_names = detector;
        descriptors_name = descriptor;
    }
    

    std::ofstream features_file;
    std::ofstream matched_feature_file;
    std::ofstream logTime_file_ms;
    if(writeFile){
        features_file.open("../csv/feature_count.csv");
        matched_feature_file.open("../csv/matched_feature_count.csv");
        logTime_file_ms.open("../csv/log_time_ms.csv");

        features_file << "Detector";
        matched_feature_file <<"Detector-Descriptor";
        logTime_file_ms<<"Detector-Descriptor";
        for (size_t imgIndex = 0; imgIndex < imgEndIndex - imgStartIndex; imgIndex++)
        {
            features_file << ","<<"Img" <<imgIndex;
            matched_feature_file << ","<<"Img"<<imgIndex<<"-Img"<<imgIndex+1;
            logTime_file_ms<< ","<<"Img"<<imgIndex<<"-Img"<<imgIndex+1;
        }
        features_file << ","<<"Img" <<imgEndIndex<<std::endl;
        matched_feature_file <<std::endl;
        logTime_file_ms<<std::endl;
    }

    for (auto detectorType:detectors_names) {
        bool written = false;
        for(auto descriptorName:descriptors_name) {
            // AKAZE works only with AKAZE
            if((detectorType.compare("AKAZE") != 0)&&(descriptorName.compare("AKAZE")==0))
               continue;
            // SIFT doesnot work with ORB
            if((detectorType.compare("SIFT") == 0)&&(descriptorName.compare("ORB")==0))
               continue;
            
            // clear the buffer for next iteration
            dataBuffer.clear();

            std::string descriptorType;
            if ((descriptorName.compare("SIFT")==0) || (descriptorName.compare("SURF")==0)) {
                descriptorType = "DES_HOG";
            }
            else{
                descriptorType = "DES_BINARY";
            }

            if(writeFile) {
                if(!written)
                    features_file << detectorType ;
                matched_feature_file <<detectorType <<"+"<< descriptorName;
                logTime_file_ms<<detectorType <<"+"<< descriptorName;
            }
            
            string matcherType = "MAT_BF";          // MAT_BF, MAT_FLANN
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            /* MAIN LOOP OVER ALL IMAGES */
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                //now push the data into buffer
                dataBuffer.push_back(frame);
                // remove image at index 0 if size = dataBufferSize
                if (dataBuffer.size() == dataBufferSize+1)
                    dataBuffer.erase(dataBuffer.begin());
                
    
                //// EOF STUDENT ASSIGNMENT
                cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image
                //string detectorType = "BRISK";

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE,  SIFT. SURF
                double time_det = 0.0 ;
                bVis = false;
                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, time_det, bVis);
                }
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, time_det, bVis);
                }
                else
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, time_det, bVis);
                }
                bVis = false;
                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);
                if (bFocusOnVehicle)
                {
                    int i = 0;
                    for (auto it = keypoints.begin(); it != keypoints.end(); ) {
                        if (!vehicleRect.contains(it->pt)){
                            keypoints.erase(it); // if erased, it goes to it++
                        }
                        else
                            ++it;
                    }
                    std::cout << " Number of " << detectorType << " keypoints inside rectangle " << keypoints.size() <<std::endl;
                    if(writeFile && !written) {
                        features_file << ","<< keypoints.size();
                    }

                    /*
                    {
                    cv::Mat visImage = img.clone();
                    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                    string windowName = detectorType + " Detector Results";
                    cv::namedWindow(windowName, 6);
                    imshow(windowName, visImage);
                    cv::waitKey(0);
                    }
                    */
                }

                //// EOF STUDENT ASSIGNMENT

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;
                cout << "#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;
                //string descriptorName = "SIFT"; // BRIEF, ORB, FREAK, AKAZE, SIFT
                double time_desc = 0.0;
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorName, time_desc);
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {

                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    //string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    //string descriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
                    //string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
                    double time_match = 0;
                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                    matches, descriptorType, matcherType, selectorType,time_match);

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    if(writeFile) {
                        matched_feature_file << ","<< matches.size();
                        logTime_file_ms<< ","<<(time_det+time_desc+time_match) * 1000;
                    }

                    cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                }

            } // eof loop over all images
            if(writeFile) {
                matched_feature_file <<std::endl;
                logTime_file_ms<<std::endl;
                written = true;
            }
        }
        if(writeFile) {
            features_file << std::endl;
            matched_feature_file <<std::endl;
            logTime_file_ms<<std::endl;
        }
    }
    if(writeFile) {
        //close all files
        features_file.close();
        matched_feature_file.close();
        logTime_file_ms.close();
    }
    return 0;
}
