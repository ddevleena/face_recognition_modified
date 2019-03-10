import face_recognition
import cv2

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture of JUSTIN and learn how to recognize it.
justin_image = face_recognition.load_image_file("justin.jpg")
justin_face_encoding = face_recognition.face_encodings(justin_image)[0]

# Load a sample of DEVLEENA picture and learn how to recognize it.
devleena_image = face_recognition.load_image_file("Devleena.jpeg")
devleena_face_encoding = face_recognition.face_encodings(devleena_image)[0]

# Load a sample picture of LANDAN and learn how to recognize it.
landan_image = face_recognition.load_image_file("landan.jpg")
landan_face_encoding = face_recognition.face_encodings(landan_image)[0]

# Load a sample picture of ALBERTO and learn how to recognize it.
alberto_image = face_recognition.load_image_file("alberto.jpeg")
alberto_face_encoding = face_recognition.face_encodings(alberto_image)[0]

# Load a sample picture of ADAM and learn how to recognize it.
adam_image = face_recognition.load_image_file("adam.jpeg")
adam_face_encoding = face_recognition.face_encodings(adam_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    devleena_face_encoding,
    justin_face_encoding,
    landan_face_encoding,
    alberto_face_encoding,
    adam_face_encoding
]
known_face_names = [
    "Devleena",
    "justin",
    "landan",
    "alberto",
    "adam"
]


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
