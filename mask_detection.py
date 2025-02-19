import numpy as np
import imutils
import cv2
import pathlib
import sys
import asyncio
from pydub import AudioSegment
from pydub.playback import play
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream


async def please_wear_mask_gr():
	sound_file = pathlib.Path("sounds").joinpath("please_wear_mask_gr.wav")
	sound = AudioSegment.from_file(sound_file)
	play(sound)


def detect_and_predict_mask(frame, faceNet, maskNet):
	"""
	Given a frame from the video feed and a NN for the face and mask, predicts if the
	user is wearing a mask

	:param frame:
	:param faceNet:
	:param maskNet:
	:return:
	"""

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of face locations and their corresponding location
	return (locs, preds)


# load the serialized face detector model
prototxtPath = pathlib.Path("face_detector").joinpath("deploy.prototxt")
weightsPath = pathlib.Path("face_detector").joinpath("res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(str(prototxtPath), str(weightsPath))

# load the face mask detector model from disk
maskNet = load_model(pathlib.Path("model").joinpath("mask_detector.model"))

# initialize the video stream
sys.stdout.write("[INFO] starting video stream...\n")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=900)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	label = ""

	# loop over the detected face and surroundings
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label_perc = f"{label}: {round(max(mask, withoutMask) * 100, 2)}%"

		# display the label and bounding box rectangle on the live video
		cv2.putText(frame, label_perc, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	"""if label == "No Mask":
		loop = asyncio.get_event_loop()
		loop.run_until_complete(please_wear_mask_gr())"""

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
