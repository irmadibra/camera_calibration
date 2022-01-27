#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

#define A4_WIDTH 297 // mm
#define A4_HEIGHT 210 // mm



struct t_calculation_parameters{
    std::vector<cv::Point2f>chessboard_pixel_coordinates;
    std::string window_name;
    cv::Rect calculate_button;
    double chessboard_edge;
    cv::Size pattern_size;
    cv::Mat image;
};
// DEFINE FUNCTIONS BEGIN

void draw_axis(cv::Mat &src_, cv::Mat &dst_, std::vector<cv::Point2f> &img_points, std::vector<cv::Point2f> &corners);
void define_3d_world_point(t_calculation_parameters params, std::vector<cv::Point3f>& object_points);
void get_camera_position_world_coord(t_calculation_parameters params, const cv::Mat& R, const cv::Mat& tvec);
void calculate_camera_position(int state, void * window_parameters);
void mouse_input(int event, int x, int y, int flags, void * params);

// DEFINITION END

int main(int argc, char *argv[])
{

    const std::string window_name = "Bilberry";
    cv::Mat image = cv::imread("calibration.jpg");

    t_calculation_parameters window_parameters{
        {},
        window_name,
        cv::Rect{0,0,1000, 100},
        20, // 20 mm
        cv::Size(9, 6),
        image
    };

    image(window_parameters.calculate_button) = cv::Vec3b(200,200,200);
    cv::putText(image(window_parameters.calculate_button), "Calculate Camera Position",
                                                            cv::Point(0,
                                                            window_parameters.calculate_button.height * 0.7),
                                                            cv::FONT_HERSHEY_COMPLEX,
                                                            2,
                                                            cv::Scalar(176,122,0),5, cv::LINE_AA);

    cv::namedWindow(window_name,cv::WINDOW_KEEPRATIO);

    cv::createButton("Calculate Camera Position",calculate_camera_position, &window_parameters, cv::QT_PUSH_BUTTON, 1);

    cv::setMouseCallback(window_name, mouse_input, &window_parameters);
    while(true){
        cv::imshow(window_name, image);
        if (cv::waitKey(5) == 27){
            break;
        }
    }

    return 0;
}


void draw_axis(cv::Mat &src_, cv::Mat &dst_, std::vector<cv::Point2f> &img_points, std::vector<cv::Point2f> &corners) {
  src_.copyTo(dst_);
  cv::arrowedLine(dst_, corners[0], img_points[0], cv::Scalar(0, 0, 255), 2,
                  cv::LINE_AA, 0);
  cv::arrowedLine(dst_, corners[0], img_points[1], cv::Scalar(0, 255, 0), 2,
                  cv::LINE_AA, 0);
  cv::arrowedLine(dst_, corners[0], img_points[2], cv::Scalar(255, 0, 0), 2,
                  cv::LINE_AA, 0);

  cv::arrowedLine(dst_, corners[0], (img_points[2] +  img_points[1] + img_points[0]) / 3 , cv::Scalar(255, 255, 255), 2,
                  cv::LINE_AA, 0);

}

void define_3d_world_point(t_calculation_parameters params, std::vector<cv::Point3f>& object_points){

    for (int i = 0; i < params.pattern_size.height; i++) {

        for (int j = 0; j < params.pattern_size.width; j++) {

            object_points.push_back(cv::Point3d(i * params.chessboard_edge, j * params.chessboard_edge, 0));

        }
    }
}

void get_camera_position_world_coord(t_calculation_parameters params, const cv::Mat& R, const cv::Mat& tvec){

    cv::Mat rotation_inverse;
    cv::transpose(R, rotation_inverse);

    //Get camera position in world cordinates
    cv::Mat world_position_front_cam = -rotation_inverse * tvec;

    cv::Mat row = world_position_front_cam.reshape(0,1);    // Treat as vector
    std::ostringstream os;
    os << row;                             // Put to the stream
    std::string camera_pos_text = "Camera Position:"+ os.str();
    cv::putText(params.image,camera_pos_text , cv::Point(0,250), cv::FONT_HERSHEY_COMPLEX, 1.5f, cv::Scalar(255,255,255),2, cv::LINE_AA);
}

void calculate_camera_position(int state, void * window_parameters){

    t_calculation_parameters *params = (reinterpret_cast<t_calculation_parameters*>(window_parameters));

    const cv::Mat camera_matrix = (cv::Mat1d(3, 3) << 3751.937744140625, 0, 1280,
                                                      0, 3821.9775390625, 960,
                                                      0, 0, 1 );

    const cv::Mat distortion_coefficients = (cv::Mat1d(1, 8) << -1.2093325853347778,
                                       -1.041786551475525,
                                       -0.004819112829864025,
                                       -0.0027203019708395004,
                                       64.01393127441406,
                                       -0.5568028688430786,
                                       -3.6638600826263428,
                                       78.03067016601562);

    // Defining the world coordinates for 3D points
    // z plane = 0
    std::vector<cv::Point3f> object_points, axis;

    define_3d_world_point(*params, object_points );


    axis.push_back(cv::Point3d(3 * params->chessboard_edge, 0.0, 0.0));
    axis.push_back(cv::Point3d(0.0, 3 * params->chessboard_edge, 0.0));
    axis.push_back(cv::Point3d(0.0, 0.0, -3 * params->chessboard_edge));


    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(params->image,
                                           params->pattern_size,
                                           corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
    if ( found ){

        cv::Mat gray;
        cv::cvtColor(params->image,gray,cv::COLOR_BGR2GRAY);

        cv::cornerSubPix(gray, corners,  params->pattern_size, cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

        cv::drawChessboardCorners(params->image, params->pattern_size, cv::Mat(corners), found);

    }
//    cv::Mat undistorted_image;
//    cv::undistort(params->image, undistorted_image, camera_matrix, distortion_coefficients);
//    params->image = undistorted_image;

    cv::Mat rvec, tvec;
    //estimation of camera pose
    cv::solvePnPRansac(object_points, corners, camera_matrix,
                       distortion_coefficients, rvec, tvec, false,100, 8.0, 0.99,
                       cv::noArray(), cv::SOLVEPNP_EPNP);



    std::vector<cv::Point2f> projected_points;

    cv::projectPoints(axis, rvec, tvec, camera_matrix, distortion_coefficients, projected_points, cv::noArray(), 0);
    draw_axis(params->image, params->image, projected_points, corners);

    for(unsigned int i = 0; i < projected_points.size(); ++i)
    {
        cv::circle(params->image,projected_points[i],10,cv::Scalar(0, 255, 255),10);
    }

    cv::Mat R(3, 3, CV_32F);
    cv::Rodrigues(rvec, R);

    get_camera_position_world_coord(*params, R , tvec);

}

void mouse_input(int event, int x, int y, int flags, void * params)
{
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        t_calculation_parameters *mouse_params = (reinterpret_cast<t_calculation_parameters*>(params));

        if (mouse_params->calculate_button.contains(cv::Point(x, y))){
                cv::rectangle(mouse_params->image(mouse_params->calculate_button), mouse_params->calculate_button, cv::Scalar(0,255,255), 10);
                if (mouse_params->chessboard_pixel_coordinates.size() == 4){
                    calculate_camera_position(0, mouse_params);
                }

        }else{
            if (mouse_params->chessboard_pixel_coordinates.size() <= 4){
                mouse_params->chessboard_pixel_coordinates.push_back(cv::Point2f(x,y));
                cv::circle(mouse_params->image,mouse_params->chessboard_pixel_coordinates.back(),5,cv::Scalar(0, 255, 255),5);

                if (mouse_params->chessboard_pixel_coordinates.size() == 4){
                    for( int i = 0; i < mouse_params->chessboard_pixel_coordinates.size() - 1; i++){
                        cv::Point p0 = mouse_params->chessboard_pixel_coordinates.at(i);
                        cv::Point p1 = mouse_params->chessboard_pixel_coordinates.at(i+1);
                        cv::line(mouse_params->image, p0, p1,cv::Scalar(0, 255, 0),5);
                    }

                    cv::line(mouse_params->image,
                             mouse_params->chessboard_pixel_coordinates.back(),
                             mouse_params->chessboard_pixel_coordinates.front(),
                             cv::Scalar(0, 255, 0),
                             5);

                }
            }
        }
    }
}


