## Tasks
### MP.1 Data Buffer Optimization
Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end. 
```
    DataFrame frame;
    frame.cameraImg = imgGray;
    //now push the data into buffer
    dataBuffer.push_back(frame);
    // remove image at index 0 if size = dataBufferSize
    if (dataBuffer.size() == dataBufferSize+1)
        dataBuffer.erase(dataBuffer.begin());
```

### MP.2 Keypoint Detection
Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly. </br>

In MidTermProject_Camera_Student.cpp
```
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
```
In matching2D_Student.cpp
```
    // Detect keypoints in image using the traditional Shi-Thomasi detector
    void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& dt, bool bVis)
    {
        // compute detector parameters based on image size
        int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        double maxOverlap = 0.0; // max. permissible overlap between two features in %
        double minDistance = (1.0 - maxOverlap) * blockSize;
        int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

        double qualityLevel = 0.01; // minimal accepted quality of image corners
        double k = 0.04;

        // Apply corner detection
        double t = (double)cv::getTickCount();
        vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

        // add corners to result vector
        for (auto it = corners.begin(); it != corners.end(); ++it)
        {

            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
            newKeyPoint.size = blockSize;
            keypoints.push_back(newKeyPoint);
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        dt = t;

        // visualize results
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "Shi-Tomasi Corner Detector Results";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);
        }
    }

    void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& dt, bool bVis) 
    {
            // Harris Detector parameters
        int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
        int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
        int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
        double k = 0.04;       // Harris parameter (see equation for details)

        double t = (double)cv::getTickCount();
        // Detect Harris corners and normalize output
        cv::Mat dst, dst_norm, dst_norm_scaled;
        dst = cv::Mat::zeros(img.size(), CV_32FC1);    
        cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        cv::convertScaleAbs(dst_norm, dst_norm_scaled);
        
        cv::KeyPoint kp;
        double maxOverlap = 0.0;

        for (size_t i = 0; i <img.size().height ; i++)
            for( size_t j = 0; j <img.size().width ; j++) {
                if((int)dst_norm.at<float>( i,j)  >= minResponse) {
                    kp.pt= cv::Point2f(j,i);
                    kp.response = (int)dst_norm.at<float>( i,j);
                    kp.size = 2 * apertureSize;
                    //std::cout << "corner found "<< std::endl;

                
                    bool bOverlap = false;
                    for (vector<cv::KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it) {
                        double kptOverlap = cv::KeyPoint::overlap(kp, *it);
                        if (kptOverlap > maxOverlap) {
                            bOverlap = true;
                            if( kp.response > (*it).response) {
                                *it = kp;
                                break;
                            }
                        }  
                    }
                    if (!bOverlap)
                    {                                     // only add new key point if no overlap has been found in previous NMS
                            keypoints.push_back(kp); // store new keypoint in dynamic list
                    }
                }
                
            }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        dt = t;

        // visualize results
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "Harris Corner Detector Results";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);
        }
    }


    void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, double& dt, bool bVis) {
        cv::Ptr<cv::FeatureDetector> detector;
        if (detectorType.compare("FAST") == 0) 
        {   // FAST parameters
            int threshold = 30;                                                              // difference between intensity of the central pixel and pixels of a circle around this pixel
            bool bNMS = true;                                                                // perform non-maxima suppression on keypoints
            int type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
            detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        }
        else if (detectorType.compare("BRISK") == 0) {
            detector = cv::BRISK::create();
        }
        else if (detectorType.compare("ORB") == 0){
            int nfeatures = 500;
            detector = cv::ORB::create(nfeatures);
        }
        else if (detectorType.compare("AKAZE") == 0){
            detector = cv::AKAZE::create();
        }
        else if (detectorType.compare("SIFT") == 0){
            detector = cv::xfeatures2d::SIFT::create();
        }
        else if (detectorType.compare("SURF") == 0){
            int minHessian = 400;
            detector = cv::xfeatures2d::SURF::create(minHessian);
        }
        else{
            std::cout << "No descriptor named " << detectorType <<" found "<<std::endl;
            return;
        }
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        dt = t;
        
        cout << detectorType <<" detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        // visualize results
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = detectorType + " Detector Results";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);
        }
    }
```
### MP.3 Keypoint Removal
Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing. 
```
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
    }
```
### MP.4 Keypoint Descriptors
Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly. </br>
In matching2D_Student.cpp
```
    // Use one of several types of state-of-art descriptors to uniquely identify keypoints
    void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double& dt)
    {
        // select appropriate descriptor
        cv::Ptr<cv::DescriptorExtractor> extractor;
        if (descriptorType.compare("BRISK") == 0)
        {

            int threshold = 30;        // FAST/AGAST detection threshold score.
            int octaves = 3;           // detection octaves (use 0 to do single scale)
            float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

            extractor = cv::BRISK::create(threshold, octaves, patternScale);
        }
        else if (descriptorType.compare("BRIEF") == 0) {
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        }
        else if (descriptorType.compare("ORB") == 0) {
            int nfeatures = 500;
            extractor = cv::ORB::create(nfeatures);
        }
        else if (descriptorType.compare("FREAK") == 0) {
            extractor = cv::xfeatures2d::FREAK::create();
        }
        else if (descriptorType.compare("AKAZE") == 0) {
            extractor = cv::AKAZE::create();
        }
        else if (descriptorType.compare("SIFT") == 0) {
            extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
        }
        else
        {
            std::cout << "No descriptor named " << descriptorType <<" found "<<std::endl;
            return;
        }

        // perform feature description
        double t = (double)cv::getTickCount();
        extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
        dt =t;
    }
```
### MP.5 Descriptor Matching
Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function. </br>
In matching2D_Student.cpp
```
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create(); 	
        //cout << "FLANN matching";
    }
```
```
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        int k = 2;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, k); 
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
        // TODO : filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for(auto it = knn_matches.begin(); it!=knn_matches.end(); ++it){
            if((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
                matches.push_back((*it)[0]);
        }
        //cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
```
### MP.6 Descriptor Distance Ratio
Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints. </br>
- See above.
### MP.7 Performance Evaluation 1
Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented. </br>

|Detector |Img0|Img1|Img2|Img3|Img4|Img5|Img6|Img7|Img8|Img9|Total|
|---------|---|---|---|---|---|---|---|---|---|----|-----|
|SHITOMASI|125|118|123|120|120|113|114|123|111|112 |1179 |
|HARRIS   |17 |14 |19 |22 |26 |47 |18 |33 |27 |35  |258  |
|FAST     |149|152|150|155|149|149|156|150|138|143 |1491 |
|BRISK    |264|282|282|277|297|279|289|272|266|254 |2762 |
|ORB      |92 |102|106|114|109|125|129|129|127|127 |1160 |
|AKAZE    |166|157|161|155|163|164|173|175|177|179 |1670 |
|SIFT     |138|132|124|137|134|140|137|148|159|137 |1386 |
|SURF     |157|160|158|163|160|178|194|183|166|177 |1696 |

We can see that BRISK gives the highest number of detected Keypoints while Harris provides low number of detected Keypoints.

### MP.8 Performance Evaluation 2
Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8. </br>
####  Used approach
-  BF matcher type
-  KNN selector type with distance ratio of 0.8

#### Note
- Matching could not be performed when AKAZE descriptors was uses with other detectors. Also, SIFT  descriptors donot work with ORB detectors.
-  NORM_HAMMING should be used with ORB, BRISK, BRIEF, FREAK and AKAZE. L1 and L2 Norm is used with SIFT or SURF.

|Detector-Descriptor|Img0-Img1|Img1-Img2|Img2-Img3|Img3-Img4|Img4-Img5|Img5-Img6|Img6-Img7|Img7-Img8|Img8-Img9|mean|
|---------|---|---|---|---|---|---|---|---|---|----|
|SHITOMASI+BRISK|95 |88 |80 |90 |82 |79 |85 |86 |82 |85.22222222222223|
|SHITOMASI+SIFT|112|109|104|103|99 |101|96 |106|97 |103 |
|SHITOMASI+BRIEF|115|111|104|101|102|102|100|109|100|104.88888888888889|
|SHITOMASI+ORB|106|102|99 |102|103|97 |98 |104|97 |100.88888888888889|
|SHITOMASI+FREAK|86 |90 |86 |88 |86 |80 |81 |86 |85 |85.33333333333333|
|HARRIS+BRISK|12 |10 |14 |16 |16 |17 |15 |22 |21 |15.88888888888889|
|HARRIS+SIFT|14 |11 |16 |20 |21 |23 |13 |24 |23 |18.333333333333332|
|HARRIS+BRIEF|14 |11 |16 |21 |23 |28 |16 |25 |24 |19.77777777777778|
|HARRIS+ORB|12 |12 |15 |19 |23 |21 |15 |25 |23 |18.333333333333332|
|HARRIS+FREAK|13 |12 |15 |16 |16 |21 |12 |21 |19 |16.11111111111111|
|FAST+BRISK|97 |104|101|98 |85 |107|107|100|100|99.88888888888889|
|FAST+SIFT|118|123|110|119|114|119|123|117|103|116.22222222222223|
|FAST+BRIEF|119|130|118|126|108|123|131|125|119|122.11111111111111|
|FAST+ORB |118|123|112|126|106|122|122|123|119|119 |
|FAST+FREAK|98 |99 |91 |98 |85 |99 |102|101|105|97.55555555555556|
|BRISK+BRISK|171|176|157|176|174|188|173|171|184|174.44444444444446|
|BRISK+SIFT|182|193|169|183|171|195|194|176|183|182.88888888888889|
|BRISK+BRIEF|178|205|185|179|183|195|207|189|183|189.33333333333334|
|BRISK+ORB|162|175|158|167|160|182|167|171|172|168.22222222222223|
|BRISK+FREAK|160|177|155|173|161|183|169|178|168|169.33333333333334|
|ORB+BRISK|73 |74 |79 |86 |79 |92 |89 |88 |91 |83.44444444444444|
|ORB+SIFT |67 |79 |78 |79 |82 |95 |94 |94 |94 |84.66666666666667|
|ORB+BRIEF|49 |43 |45 |59 |53 |78 |67 |84 |65 |60.333333333333336|
|ORB+ORB  |67 |70 |72 |84 |91 |101|91 |93 |91 |84.44444444444444|
|ORB+FREAK|42 |36 |44 |47 |44 |51 |52 |48 |56 |46.666666666666664|
|AKAZE+BRISK|137|125|129|129|131|132|142|146|144|135 |
|AKAZE+SIFT|134|134|130|136|137|147|147|154|151|141.11111111111111|
|AKAZE+BRIEF|141|134|131|130|134|146|150|148|152|140.66666666666666|
|AKAZE+ORB|131|129|127|117|130|131|137|135|145|131.33333333333334|
|AKAZE+FREAK|126|129|127|121|122|133|144|147|138|131.88888888888889|
|AKAZE+AKAZE|138|138|133|127|129|146|147|151|150|139.88888888888889|
|SIFT+BRISK|64 |66 |62 |66 |59 |64 |64 |67 |80 |65.77777777777777|
|SIFT+SIFT|82 |81 |85 |93 |90 |81 |82 |102|104|88.88888888888889|
|SIFT+BRIEF|86 |78 |76 |85 |69 |74 |76 |70 |88 |78  |
|SIFT+FREAK|65 |72 |64 |66 |59 |59 |64 |65 |79 |65.88888888888889|
|SURF+BRISK|117|126|119|118|110|137|137|129|126|124.33333333333333|
|SURF+SIFT|121|118|129|125|119|148|151|152|145|134.22222222222223|
|SURF+BRIEF|121|133|124|134|119|147|146|141|142|134.11111111111111|
|SURF+ORB |116|120|113|121|109|133|149|131|125|124.11111111111111|
|SURF+FREAK|107|112|102|103|95 |117|117|117|120|110 |

### MP.9 Performance Evaluation 3
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles. </br>
- Time inside each cells represent time taken for feature detection,  descriptor  extraction and feature matching. </br>

|Detector-Descriptor|Img0-Img1|Img1-Img2|Img2-Img3|Img3-Img4|Img4-Img5|Img5-Img6|Img6-Img7|Img7-Img8|Img8-Img9|mean|
|---------|---|---|---|---|---|---|---|---|---|----|
|SHITOMASI+BRISK|22.9894|23.5323|22.5496|22.622700000000002|23.2382|22.6947|22.328|27.5874|27.2898|23.87023333333333|
|SHITOMASI+SIFT|42.6522|52.1742|54.7993|54.9128|50.9523|52.7911|53.3256|54.7531|53.4555|52.20178888888889|
|SHITOMASI+BRIEF|26.4238|21.0097|25.5046|24.5229|18.149|25.5942|21.6358|24.2585|23.9518|23.450033333333334|
|SHITOMASI+ORB|25.046|27.4032|25.7164|22.5296|27.2018|27.5276|23.2032|25.0467|26.7192|25.599300000000003|
|SHITOMASI+FREAK|101.036|91.1893|91.0023|90.8469|91.8115|90.8865|91.012|90.9755|91.4425|92.24472222222224|
|HARRIS+BRISK|22.624|23.7129|25.2486|26.3485|54.1943|24.2956|28.3922|25.7331|34.9457|29.499433333333336|
|HARRIS+SIFT|43.319|48.0394|41.7834|44.5574|80.4019|41.3387|47.3565|50.7345|59.2018|50.74806666666667|
|HARRIS+BRIEF|24.2906|20.9505|20.9843|21.7853|50.2821|20.7916|24.4021|22.0121|31.4727|26.330144444444446|
|HARRIS+ORB|23.6972|23.8562|23.6145|26.8469|56.5943|26.6299|27.4949|24.5596|33.6059|29.655488888888883|
|HARRIS+FREAK|99.8878|98.5079|98.8801|97.72710000000001|127.171|97.3426|102.63|102.088|110.725|103.88438888888888|
|FAST+BRISK|4.67952|4.21545|4.75608|4.66341|4.70918|4.3681|4.30933|4.08942|4.28906|**4.453283333333333**|
|FAST+SIFT|28.3038|31.344|27.2418|31.2794|27.5772|29.5671|29.2621|29.1088|28.5408|29.136111111111113|
|FAST+BRIEF|3.18157|2.72376|2.37842|2.67106|3.47467|2.72021|2.60569|2.30561|2.62613|**2.7430133333333337**|
|FAST+ORB |3.89844|4.42763|4.0519|3.87398|4.33153|4.045|3.96317|5.45803|3.89916|**4.216537777777778**|
|FAST+FREAK|78.741|78.1594|78.257|78.2936|78.4432|78.3379|77.945|78.4101|77.8439|78.27012222222221|
|BRISK+BRISK|76.9227|77.9054|75.8613|76.574|76.0805|75.5383|73.8455|74.3102|74.8123|75.76113333333333|
|BRISK+SIFT|119.047|119.431|118.9|119.701|118.726|119.229|116.575|117.916|120.393|118.87977777777779|
|BRISK+BRIEF|74.0212|71.9209|71.0247|72.7183|72.2539|72.3166|71.3238|71.8569|71.8988|72.14834444444445|
|BRISK+ORB|81.6385|80.8236|79.9293|80.8534|81.2238|79.5507|79.06|79.8683|80.0574|80.33388888888891|
|BRISK+FREAK|150.425|148.605|149.015|150.648|149.023|148.727|147.816|150.38|148.436|149.23055555555553|
|ORB+BRISK|13.7188|13.8606|13.9183|13.6037|14.1473|13.1819|14.2196|14.215|14.5836|13.938755555555556|
|ORB+SIFT |66.3877|62.7979|64.4502|62.0841|62.3585|67.3113|66.3459|70.9212|66.9525|65.51214444444444|
|ORB+BRIEF|12.1141|12.779|13.981300000000001|12.9673|11.9756|11.7446|13.169|14.181|12.7957|12.856399999999999|
|ORB+ORB  |22.4974|22.6735|23.2659|23.2309|22.2981|21.9879|22.1283|22.6272|22.2621|22.552366666666664|
|ORB+FREAK|87.5458|87.3884|87.6115|87.3176|87.2396|87.7159|87.2576|87.9183|88.1321|87.56964444444446|
|AKAZE+BRISK|83.3638|80.9752|82.0266|82.8786|85.9377|83.921|85.5284|87.3415|82.2534|83.8029111111111|
|AKAZE+SIFT|118.677|111.864|123.867|114.555|107.071|109.135|105.074|108.567|107.152|111.77355555555556|
|AKAZE+BRIEF|73.5663|70.9324|72.7417|74.8104|75.2798|74.2766|69.5105|75.0593|72.1026|73.14217777777779|
|AKAZE+ORB|83.0513|87.1116|87.1539|87.9301|86.3889|81.9153|84.7147|84.2658|87.2458|85.53082222222221|
|AKAZE+FREAK|160.677|158.949|157.644|159.017|163.783|156.338|156.252|157.448|163.997|159.345|
|AKAZE+AKAZE|144.288|135.669|137.814|137.859|135.869|131.187|134.98|136.69|137.59|136.88288888888889|
|SIFT+BRISK|133.924|162.596|163.573|162.972|160.791|163.571|161.962|162.17|160.947|159.16733333333332|
|SIFT+SIFT|231.77|266.404|231.04|232.924|229.558|232.149|227.437|230.632|231.203|234.79077777777775|
|SIFT+BRIEF|157.429|157.907|155.585|157.872|158.268|157.595|159.477|159.479|156.775|157.8207777777778|
|SIFT+FREAK|234.033|235.205|237.127|236.577|235.992|235.439|236.713|236.034|234.993|235.7903333333333|
|SURF+BRISK|61.7243|63.0505|60.4783|61.1631|64.2189|66.1398|63.2065|62.1457|64.3473|62.941599999999994|
|SURF+SIFT|126.826|119.712|124.816|127.806|128.401|132.789|128.454|126.34|131.494|127.40422222222224|
|SURF+BRIEF|59.4528|61.8755|60.1825|61.3589|59.3047|59.4052|62.374|60.2292|61.745|60.65864444444445|
|SURF+ORB |69.4567|68.8506|66.6564|65.6788|67.8341|67.1695|65.9046|68.6489|69.1467|67.70514444444444|
|SURF+FREAK|137.852|139.208|141.122|139.235|139.514|138.898|142.74|142.016|142.623|140.35644444444446|

#### Best detector-descriptor combinations based on matched_feature_count/time (see csv/matched_VS_Time.xls)
- FAST + BRIEF
- FAST + ORB
- FAST + BRISK
