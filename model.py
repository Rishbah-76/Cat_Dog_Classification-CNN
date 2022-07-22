from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dense

#Initialize our classifier using keras
clf=Sequential()
    #Creating thr architecture
    # Step 1 - Convolution
clf.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
clf.add(Conv2D(32, (3, 3), activation = 'relu'))
    # Step 2 - Pooling
clf.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
clf.add(Conv2D(16, (3, 3), activation = 'relu'))
clf.add(Conv2D(16, (3, 3), activation = 'relu'))
    # Step 3 - Flattening
clf.add(Flatten())

    # Step 4 - Full connection
clf.add(Dense(units = 128, activation = 'relu'))
clf.add(Dense(units = 1, activation = 'sigmoid'))
    # Compiling the CNN
clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Adding Image and data Augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('data/validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = clf.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,    
                         validation_steps = 2000)

clf.save("cat_dog_model.h5")
print("Saved model to disk")
