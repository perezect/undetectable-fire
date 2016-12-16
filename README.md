# Fire Detection Algorithm
## Contributors
Felis Perez and William Chen
## Dependencies
openCV3.1.0, numPy
## Usage instructions
Simply navigate to the directory where detect.py is found and type in the following console command:

``` python detect.py [path to video input] ```

This will run our fire detection algorithm on the video input at the specified path. Our fire detection algorithm provides an output video file with frame by frame threat level assignment and room safety condition based off threat levels and fire contours detected. Threat levels range from 0 to 10, and the room safety condition can either be 'SAFE' or 'DANGER'.

## Inputs
Some inputs used to test this project are included in the src file.
