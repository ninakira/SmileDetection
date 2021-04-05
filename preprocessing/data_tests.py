from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator()

train_it = datagen.flow_from_directory('/data/augmented_celeba/train/', class_mode='binary', batch_size=128)
val_it = datagen.flow_from_directory('/data/augmented_celeba/validation/', class_mode='binary', batch_size=128)

print(train_it)
print(val_it)