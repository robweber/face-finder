import face_recognition

rob_image = face_recognition.load_image_file("pics/rob.jpg")
rob_face = face_recognition.face_encodings(rob_image)[0]

test_image = face_recognition.load_image_file("pics/training/2022-11-21 - Magic Kingdom Park - Main Street USA_7.jpeg")
test_encoding = face_recognition.face_encodings(test_image)

results = face_recognition.compare_faces(rob_face, test_encoding)
print(results)
