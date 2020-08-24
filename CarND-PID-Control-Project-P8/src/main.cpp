#include <math.h>
#include <uWS/uWS.h>
#include <iostream>
#include <string>
#include "json.hpp"
#include "PID.h"

// for convenience
using nlohmann::json;
using std::string;
static bool low_tps_ = false;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != string::npos) {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main() {
  uWS::Hub h;



  
  /**
   * TODO: Initialize the pid variable.
   */
  // PID object for controlling steer
   PID pidSteer;
    
  //pidSteer.Init(0.06, 0.0, 0.8);
  pidSteer.Init(0.050, 0.00200, 0.80);


  // PID object for controlling throttle
  PID pidThrottle;
  
  pidThrottle.Init(0.15, 0.0001, 0.75);

  

  h.onMessage([&pidSteer, &pidThrottle](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, 
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(string(data).substr(0, length));

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<string>());
          double speed = std::stod(j[1]["speed"].get<string>());
          double angle = std::stod(j[1]["steering_angle"].get<string>());
          double steer_value = 0.0;
          int max_steering_angle = 1;
          
          
          /**
           * TODO: Calculate steering value here, remember the steering value is
           *   [-1, 1].
           * NOTE: Feel free to play around with the throttle and speed.
           *   Maybe use another PID controller to control the speed!
           */
          
          pidSteer.UpdateError(cte);
          
          steer_value =  std::max(-0.5, std::min(-pidSteer.TotalError(), 0.5));

         
          
          
          // If the steering angle is beyond +/- 1 limit the steering angle
          // to +/- 0.5 to prevent sharp turns
          
          if (steer_value  > max_steering_angle)
          {
            steer_value  = 0.5;
          }
          else if (steer_value  < -max_steering_angle)
          {
            steer_value  = -0.5;
          }
          
         

         
          double throttle;
          if (fabs(cte) > 1) {
            // Go slow when the cte is high
            throttle = 0.25;
          } else {
            // Otherwise, use the inverse of the steering value as the throttle, with a max of 100
            throttle = fmin(1 / fabs(steer_value), 100);

            // Normalize the throttle value from [0, 100] to [0.45, 1.0]
            // normalized_x = ((ceil - floor) * (x - minimum))/(maximum - minimum) + floor
            throttle = ((1.0 - 0.45) * (throttle - 0.0)) / (100.0 - 0.0) + 0.45;

            // Slow down when the tps is low
            if (low_tps_) {
              throttle -= 0.2;
            }
          }

                    
          // DEBUG
          std::cout << "CTE: " << cte << " Steering Value: " << steer_value << std::endl;

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle;
          
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket message if
  }); // end h.onMessage

  
  
  
  
  
  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, 
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}