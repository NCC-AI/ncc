from ncc.dataset import prepare_data
from ncc.models import Model2D
from ncc.generators import generate_arrays_from_annotation
from ncc.history import save_history

# params
input_shape = (32, 32, 3)
batch_size = 4
nb_classes = 10
num_images_per_class = 100
steps_per_epoch = num_images_per_class*nb_classes // batch_size
epochs = 10

prepare_data(num_images_per_class)

# build and train model
model = Model2D(input_shape=input_shape, num_classes=10)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
model.summary()

history = model.fit_generator(
    generator=generate_arrays_from_annotation('data_set/train_annotation.csv',
                                              batch_size=batch_size,
                                              nb_classes=nb_classes,
                                              target_size=input_shape[:2],
                                              ),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=generate_arrays_from_annotation('data_set/test_annotation.csv',
                                                    batch_size=batch_size,
                                                    nb_classes=nb_classes,
                                                    target_size=input_shape[:2],
                                                    ),
    validation_steps=steps_per_epoch
)

save_history(history)
