# Gesture Recognition Using a Neural Network

The code in the repository is for creating a neural network for gesture recognition with the use of [Google Mediapipe Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer). It will recognize gestures from numbers 1-10 with a reliable accuracy. The model used over 4000 datapoints of hand landmarks with its corresponding gesture for train and test phases.

## How to Run

>- You can locally run the gesture recognition using the following command.
>```
>uvicorn api_call:app --reload
>```
>- You can also pull and use the gesture recognition docker container from the following command.
>```
>docker pull awesomenipun/gesture-recognition-1
>```

>- Then you can call use the following CURL command to send an image to recognize the gesture.
>```sh
>curl  -X POST \
>  'localhost:PORT//gesture-recognizer/process-image' \
>  --header 'Accept: */*' \
>  --form 'file=@/path/to/your/file'
>```
>
>Note: The PORT should be the port that you are running the recognition service.

## Acknowledgement
We would like to acknowledge the following repository for providing the implementation that was used as the basis for the gesture recognition model:

- [Gesture Recognition Implementation](https://github.com/kinivi/hand-gesture-recognition-mediapipe) by [Nikita Kiselov](https://github.com/kinivi)

We adapted and modified the code from this repository to fit the needs of our project. The original implementation was invaluable in getting started with the gesture recognition model.

The original work is licensed under the [Apache v2 license](http://www.apache.org/licenses/LICENSE-2.0)
