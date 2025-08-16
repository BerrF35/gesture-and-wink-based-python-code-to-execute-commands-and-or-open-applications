# gesture-and-wink-based-python-code-to-execute-commands-and-or-open-applications
Gesture and Wink Control

This project lets you control your Windows PC using hand gestures and winks. Built with OpenCV and MediaPipe, it uses your webcam to track your hands and face, then triggers system actions.

Features

Fist → Open Hand → Fist = Take a screenshot (Snip & Sketch).

Wink (right eye) = Take a screenshot (Snip & Sketch).

Middle Finger = Lock screen.

Index Finger Only = Open Google Chrome.

Spider-Man Gesture (index and pinky up, others folded) = Open Google Chrome.

Real-time webcam feed with drawn landmarks.
--------------------------------------------------------------------------------------------------------------------
Requirements

Python 3.8+

OpenCV

MediaPipe

NumPy
-------------------------------------------------------------------------------------------------------------------
Install dependencies with:

pip install opencv-python mediapipe numpy

Usage
---------------------------------------------------------------------------------------------------------------------
Clone this repo:

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

--------------------------------------------------------------------------------------------------------------------
Run the script:

python handgestureoff.py


Make gestures or wink in front of the webcam to trigger actions.

Press Q to quit the application.
----------------------------------------------------------------------------------------------------------------------
Notes

Tested on Windows (uses os.system("start ...") and rundll32.exe).

Edit the system commands in handgestureoff.py to open other applications if desired.

Works best with good lighting and a 720p+ webcam.
----------------------------------------------------------------------------------------------------------------------
install the dependencies easily before running the script by executing the given command below

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

---------------------------------------------------------------------------------------------------------------------
License

This project is licensed under the MIT License
