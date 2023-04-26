from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import ImageSerializer
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import  datasets, layers, models


class ImageRecognition(APIView):
    def post(self, request):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            (training_images, training_labels), (testing_images, testing_labels) = datasets.cirfar10.load_data()
			training_images, testing_images = training_images / 255, testing_images / 255

			class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

			training_images = training_images[:20000]
			training_labels = training_labels[:20000]
			testing_images = testing_images[:4000]
			testing_labels = testing_labels[:4000]

			model = models.Sequential()
			model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32,3)))
			model.add(layers.MaxPooling2D((2,2)))
			model.add(layers.Conv2D(64, (3,3), activation='relu'))
			model.add(layers.MaxPooling2D((2,2)))
			model.add(layers.Conv2D(64, (3,3), activation='relu'))
			model.add(layers.Flatten())
			model.add(layers.Dense(64, activation='relu'))
			model.add(layers.Dense(10, activation='softmax'))

			model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

			model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
            # Access the uploaded file using serializer.validated_data['image']
            # Perform image recognition operations here
            return Response({'status': 'success'})
        else:
            return Response(serializer.errors, status=400)
