# Description
![Using of digit recognition](Examples/Digit_recognition.gif)

This applicationn is used to recognize hadnwritten digits. I used the **mnist dataset** to train CNN. 
The application can be used to recognize digits from videostream or from image. Instead of standard webcam 
i use the Android application **Ip Webcam** that allows me to create local server and use my phone like webcam.
# Using
1. Firstly create folder, open it in IDE or text editor and clone reposytory (or download zip) 
> git clone https://github.com/vovaklh/Real-time-digit-recognition-using-CNN.git
2. Secondly install needed libraries 
> pip install -r requirements.txt
3. To recognize digits from image use (threshold and model are not required)
> python Digit_recognition.py -i path_to_image -t your_threshold -m your_model
4. To recognize digits from video use 
> python Digit_recognition.py -u url_of_server -t your_threshold -m your_model