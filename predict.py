from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True, help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True, help="path to our input video")
ap.add_argument("-o", "--output", required=True, help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128, help="size of queue for averaging")
args = vars(ap.parse_args())

LABELS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F09", "F10", "F11","F12","F13","F15", "F18"]
A=[[0] for i in range(len(LABELS))]
print(A);
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)
	##show predictions
	for i in range(len(preds)):
		A[i].append(preds[i])
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]
	text = "Config: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
	writer.write(output)

	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

print("[INFO] cleaning up...")
j=0
for k in range(len(A[0])//100):
	plt.style.use("ggplot")
	plt.figure()

	for i in range(len(A)):
		plt.plot(np.arange(100*k, (100*k)+100), A[i][100*k:(100*k)+100], label=LABELS[i])

	plt.title("Probs for each configuration")
	plt.xlabel("frame de: "+str(100*k) +" a: "+str((100*k)+100))
	plt.ylabel("prob")
	plt.legend(loc="lower left")
	plt.savefig("plot"+str(k)+".png")
	j=k


plt.style.use("ggplot")
plt.figure()
for i in range(len(A)):
	plt.plot(np.arange(100 * j, len(A[i])), A[i][100 * k:len(A[i])], label=LABELS[i])
plt.title("Probs for each configuration")
plt.xlabel("frame de: "+ str(100*j) +" jusqu a la fin")
plt.ylabel("prob")
plt.legend(loc="lower left")
plt.savefig("plot"+str(j)+".png")

writer.release()
vs.release()


