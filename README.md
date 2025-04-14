# Pose-estimation-demonstrator
The case elaboration of making a demonstrator for the pose estimation demonstrator of semester "Innovate" at Zuyd University


## Introduction
The demonstrator made by caseteam Two Pair at Zuyd University was a case for the 1st year students.
The goal of this project was to intrigue and interest visitors of the open day about the concept of pose estimation, this has been done by implemting pose estimation with a game where the player has to avoid the red boxes and touch the green ones for points.
The speed of the boxes is increased in correspondence with the amount of points the player has gathered and increases exponentionally.

## Credits
The pose estimation code is based of the following page:
https://gist.github.com/lanzani/f85175d8fbdafcabb7d480dd1bb769d9

The used models are found on the following page:
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

## Requirements

- Vistual Studio Code
- Python 3.7+
- pip
- Git

### Minimum System Requirements

| Component | Minimum Requirement |
|-----------|---------------------|
| **CPU**   | Intel i5 8th Gen or AMD Ryzen 5 2500U |
| **RAM**   | 8 GB                |
| **GPU**   | Integrated GPU (dedicated GPU recommended for better performance) |
| **Storage** | SSD (at least 2 GB free) |
| **OS**    | Windows 10 / macOS 10.15+ / Ubuntu 20.04+ |
| **Camera** | HD webcam (720p minimum) |

### Recommended System Requirements

| Component | Recommended |
|-----------|-------------|
| **CPU**   | Intel i7 (10th Gen+) or Ryzen 7 |
| **RAM**   | 16 GB                |
| **GPU**   | NVIDIA GTX 1660 or better (for GPU-accelerated inference) |
| **Storage** | SSD with 5+ GB free |
| **OS**    | Same as above |
| **Camera** | 1080p webcam |

### The installation of the program

1. Open Visual Studio Code
2. Open a new window
3. Press on "Clone from repository"
4. Enter the following url: https://github.com/TheCyberLepricon/Pose-estimation-demonstrator.git
5. Create a new folder and give it a name to your liking
6. Make sure you are in the chosen folder
7. Open "app.py"
8. In the topmenu of Visual Studio Code, select "Terminal", then select "New Terminal"
9. Put in the appeared terminal: "pip install mediapipe opencv-python". There can occur a warning, if this is the case see subscetion "Errors" below.
10. When that is installed press "Run python file" in the top right corner.

It is recommended that the full version of the pose estimation model is used, for as this model gives the most real-time accuracy compared to the lite version. However the lite version is functional, but less accurate and some faulty registrations may occur.
If this change seems as the only option for your system, change the path on line 13 to: model_path = "models/pose_landmarker_lite.task"

### Errors
This section contains solutions to the known errors that might occur in the installation of the program. This guide assumes english is the primary language of your computer, if this is not the case some searches will not turn up result, as a fix look up the translation of the searches.
#### Directory not on PATH
This is how the error could look like:
![image](https://github.com/user-attachments/assets/fc027836-a46f-46ec-816b-20662fd2c424)

To solve this follow the next steps:
1. Open the start menu of your computer
2. Type "environment variables"
3. Select: "Edit the system environment variables"
4. Open Environment Variables Window
5. Edit Path for your user account; Under "User Variables", find the variable named "Path" and select it
6. Click "Edit"
7. Click "New"
8. Paste the your directory path that occured in the warning.
9. Click "OK" to close each dialog box
10. Close any open terminals and restart them to apply the changes


