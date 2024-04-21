import sys
import time
import customtkinter
import tkinter as Tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import random
import shutil
import zipfile
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from kaggle.api.kaggle_api_extended import KaggleApi
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
import pandas as pd


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


local_zip = "./fruits-fresh-and-rotten-for-classification.zip"
if not os.path.exists(local_zip):
    print("Downloading Dataset from Kaggle...")
    print('Download Size: 3.58 GB')
    print("This process might take 10-15 mins.")
    print("Answer the dialog box to continue...")

    # Display a dialog box with a message and yes/no buttons
    result = Tk.messagebox.askyesno("Downloading Dataset from Kaggle", "Download Size: 3.58 GB. This process might "
                                                                       "take 10-15 mins. Wish to continue?")

    if result:
        # User selected 'Yes', proceed with the download
        print("User chose to continue.")
        # Add your download code here
    else:
        # User selected 'No', do not proceed with the download
        Tk.messagebox.showinfo("Download Cancelled", "Exiting the program!")
        print("User chose not to continue.")
        exit()

    api = KaggleApi()
    api.dataset_download_files(dataset="sriramr/fruits-fresh-and-rotten-for-classification", path="./")
    print("Dataset download complete!")

    print("Extracting Dataset...")
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('./tmp')
    zip_ref.close()
    print("Dataset extraction complete!")


    def make_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)
            return path
        else:
            shutil.rmtree(path)
            os.mkdir(path)
            return path


    print("Creating Directories...")
    try:
        base_dir = './tmp'
        fruit_dir = make_dir(os.path.join(base_dir, 'fruit-dataset'))
        train_dir = make_dir(os.path.join(fruit_dir, 'train'))
        validation_dir = make_dir(os.path.join(fruit_dir, 'val'))
        test_dir = make_dir(os.path.join(fruit_dir, 'test'))
        preview_dir = make_dir(os.path.join(fruit_dir, 'preview'))

        train_fresh_apples_dir = make_dir(os.path.join(train_dir, 'Fresh Apples'))
        train_fresh_bananas_dir = make_dir(os.path.join(train_dir, 'Fresh Bananas'))
        train_fresh_oranges_dir = make_dir(os.path.join(train_dir, 'Fresh Oranges'))
        train_rotten_apples_dir = make_dir(os.path.join(train_dir, 'Rotten Apples'))
        train_rotten_bananas_dir = make_dir(os.path.join(train_dir, 'Rotten Bananas'))
        train_rotten_oranges_dir = make_dir(os.path.join(train_dir, 'Rotten Oranges'))

        validation_fresh_apples_dir = make_dir(os.path.join(validation_dir, 'Fresh Apples'))
        validation_fresh_bananas_dir = make_dir(os.path.join(validation_dir, 'Fresh Bananas'))
        validation_fresh_oranges_dir = make_dir(os.path.join(validation_dir, 'Fresh Oranges'))
        validation_rotten_apples_dir = make_dir(os.path.join(validation_dir, 'Rotten Apples'))
        validation_rotten_bananas_dir = make_dir(os.path.join(validation_dir, 'Rotten Bananas'))
        validation_rotten_oranges_dir = make_dir(os.path.join(validation_dir, 'Rotten Oranges'))

        test_fresh_apples_dir = make_dir(os.path.join(test_dir, 'Fresh Apples'))
        test_fresh_bananas_dir = make_dir(os.path.join(test_dir, 'Fresh Bananas'))
        test_fresh_oranges_dir = make_dir(os.path.join(test_dir, 'Fresh Oranges'))
        test_rotten_apples_dir = make_dir(os.path.join(test_dir, 'Rotten Apples'))
        test_rotten_bananas_dir = make_dir(os.path.join(test_dir, 'Rotten Bananas'))
        test_rotten_oranges_dir = make_dir(os.path.join(test_dir, 'Rotten Oranges'))

    except OSError:
        print("OSError")
    print("Directories created successfully!")


    def split_data(SOURCE='', TRAINING='', VALIDATION='', SPLIT_SIZE=0):
        data = os.listdir(SOURCE)
        random_data = random.sample(data, len(data))

        train_size = len(data) * SPLIT_SIZE

        for i, filename in enumerate(random_data):
            filepath = os.path.join(SOURCE, filename)
            if os.path.getsize(filepath) > 0:
                if i < train_size:
                    copyfile(filepath, os.path.join(TRAINING, filename))
                    # img = Image.open(os.path.join(TRAINING, filename)).convert('L')
                    # img.save(os.path.join(TRAINING, filename))
                else:
                    copyfile(filepath, os.path.join(VALIDATION, filename))
                    # img = Image.open(os.path.join(VALIDATION, filename)).convert('L')
                    # img.save(os.path.join(VALIDATION, filename))

    dataset_train_dir = './tmp/dataset/train'
    dataset_test_dir = './tmp/dataset/test'

    fapples_train_dir = os.path.join(dataset_train_dir, 'freshapples')
    fbananas_train_dir = os.path.join(dataset_train_dir, 'freshbanana')
    foranges_train_dir = os.path.join(dataset_train_dir, 'freshoranges')
    rapples_train_dir = os.path.join(dataset_train_dir, 'rottenapples')
    rbananas_train_dir = os.path.join(dataset_train_dir, 'rottenbanana')
    roranges_train_dir = os.path.join(dataset_train_dir, 'rottenoranges')

    fapples_test_dir = os.path.join(dataset_test_dir, 'freshapples')
    fbananas_test_dir = os.path.join(dataset_test_dir, 'freshbanana')
    foranges_test_dir = os.path.join(dataset_test_dir, 'freshoranges')
    rapples_test_dir = os.path.join(dataset_test_dir, 'rottenapples')
    rbananas_test_dir = os.path.join(dataset_test_dir, 'rottenbanana')
    roranges_test_dir = os.path.join(dataset_test_dir, 'rottenoranges')

    print('fapples_train images = ', len(os.listdir(fapples_train_dir)))
    print('fbananas_train images = ', len(os.listdir(fbananas_train_dir)))
    print('foranges_train images = ', len(os.listdir(foranges_train_dir)))
    print('rapples_train images = ', len(os.listdir(rapples_train_dir)))
    print('rbananas_train images = ', len(os.listdir(rbananas_train_dir)))
    print('roranges_train images = ', len(os.listdir(roranges_train_dir)))
    print()
    print('fapples_test images = ', len(os.listdir(fapples_test_dir)))
    print('fbananas_test images = ', len(os.listdir(fbananas_test_dir)))
    print('foranges_test images = ', len(os.listdir(foranges_test_dir)))
    print('rapples_test images = ', len(os.listdir(rapples_test_dir)))
    print('rbananas_test images = ', len(os.listdir(rbananas_test_dir)))
    print('roranges_test images = ', len(os.listdir(roranges_test_dir)))

    print("Splitting DATA into Train and Validation...")

    SPLIT_SIZE = 0.67
    split_data(fapples_train_dir, train_fresh_apples_dir, validation_fresh_apples_dir, SPLIT_SIZE)
    split_data(fbananas_train_dir, train_fresh_bananas_dir, validation_fresh_bananas_dir, SPLIT_SIZE)
    split_data(foranges_train_dir, train_fresh_oranges_dir, validation_fresh_oranges_dir, SPLIT_SIZE)
    split_data(rapples_train_dir, train_rotten_apples_dir, validation_rotten_apples_dir, SPLIT_SIZE)
    split_data(rbananas_train_dir, train_rotten_bananas_dir, validation_rotten_bananas_dir, SPLIT_SIZE)
    split_data(roranges_train_dir, train_rotten_oranges_dir, validation_rotten_oranges_dir, SPLIT_SIZE)

    SPLIT_SIZE = 1.0
    split_data(fapples_test_dir, test_fresh_apples_dir, validation_fresh_apples_dir, SPLIT_SIZE)
    split_data(fbananas_test_dir, test_fresh_bananas_dir, validation_fresh_bananas_dir, SPLIT_SIZE)
    split_data(foranges_test_dir, test_fresh_oranges_dir, validation_fresh_oranges_dir, SPLIT_SIZE)
    split_data(rapples_test_dir, test_rotten_apples_dir, validation_rotten_apples_dir, SPLIT_SIZE)
    split_data(rbananas_test_dir, test_rotten_bananas_dir, validation_rotten_bananas_dir, SPLIT_SIZE)
    split_data(roranges_test_dir, test_rotten_oranges_dir, validation_rotten_oranges_dir, SPLIT_SIZE)

    print("Splitting DATA into Train and Validation completed!")

    print(len(os.listdir('./tmp/fruit-dataset/train/Fresh Apples/')))
    print(len(os.listdir('./tmp/fruit-dataset/train/Fresh Bananas/')))
    print(len(os.listdir('./tmp/fruit-dataset/train/Fresh Oranges/')))
    print(len(os.listdir('./tmp/fruit-dataset/train/Rotten Apples/')))
    print(len(os.listdir('./tmp/fruit-dataset/train/Rotten Bananas/')))
    print(len(os.listdir('./tmp/fruit-dataset/train/Rotten Oranges/')))
    print()
    print(len(os.listdir('./tmp/fruit-dataset/val/Fresh Apples/')))
    print(len(os.listdir('./tmp/fruit-dataset/val/Fresh Bananas/')))
    print(len(os.listdir('./tmp/fruit-dataset/val/Fresh Oranges/')))
    print(len(os.listdir('./tmp/fruit-dataset/val/Rotten Apples/')))
    print(len(os.listdir('./tmp/fruit-dataset/val/Rotten Bananas/')))
    print(len(os.listdir('./tmp/fruit-dataset/val/Rotten Oranges/')))
    print()
    print(len(os.listdir('./tmp/fruit-dataset/test/Fresh Apples/')))
    print(len(os.listdir('./tmp/fruit-dataset/test/Fresh Bananas/')))
    print(len(os.listdir('./tmp/fruit-dataset/test/Fresh Oranges/')))
    print(len(os.listdir('./tmp/fruit-dataset/test/Rotten Apples/')))
    print(len(os.listdir('./tmp/fruit-dataset/test/Rotten Bananas/')))
    print(len(os.listdir('./tmp/fruit-dataset/test/Rotten Oranges/')))

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.2,  # 0.2, 0.5
        height_shift_range=0.2,  # 0.2, 0.5
        shear_range=0.2,
        zoom_range=[0.5, 1.0],  # 0.2, 0.5, [0.5,1.0]
        rotation_range=90,  # 20, 40, 60, 90
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'  # nearest, reflect, wrap
    )

    validation_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    path_aug = os.path.join(train_fresh_apples_dir, os.listdir(train_fresh_apples_dir)[-1])
    img_augmentation = image.load_img(path_aug)
    x_aug = image.img_to_array(img_augmentation)
    x_aug = x_aug.reshape((1,) + x_aug.shape)

    i = 0
    for batch in train_datagen.flow(x_aug, batch_size=1, save_to_dir=preview_dir, save_prefix='fruit',
                                    save_format='jpeg'):
        i += 1
        if i >= 20:
            break

    preview_img = os.listdir(preview_dir)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 15))
    for n in range(len(preview_img)):
        plt.subplot((int)(len(preview_img) / 4) + 1, 4, n + 1)
        plt.subplots_adjust(hspace=0.3)
        plt.imshow(image.load_img(os.path.join(preview_dir, preview_img[n]),
                                  color_mode="rgb",
                                  target_size=(150, 150),
                                  interpolation="nearest"))
        plt.axis('off')
    plt.show()

    for fn in preview_img:
        os.system(f'del {os.path.join(preview_dir, fn)}')


    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.98):
                print("\nReached 98% accuracy. Stop Training")
                self.model.stop_training = True


    callbacks = myCallback()

    # if you want to use tranfer learning, skip this cell
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    x = layers.Flatten()(pre_trained_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(6, activation='softmax')(x)
    model = Model(pre_trained_model.input, x)
    model.compile(optimizer='adam',  # RMSprop(lr=0.0001), adam
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    pd.set_option('max_colwidth', None)
    layers = [(layer, layer.name, layer.trainable) for layer in pre_trained_model.layers]
    pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

    train_len = 0
    for foldername in os.listdir('./tmp/fruit-dataset/train'):
        train_len = train_len + len(os.listdir(os.path.join('./tmp/fruit-dataset/train', foldername)))

    val_len = 0
    for foldername in os.listdir('./tmp/fruit-dataset/val'):
        val_len = val_len + len(os.listdir(os.path.join('./tmp/fruit-dataset/val', foldername)))

    print(train_len)
    print(val_len)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=32,
                                                        color_mode="rgb",
                                                        # shuffle = False,
                                                        target_size=(150, 150),  # ?
                                                        class_mode='categorical')

    validation_generator = train_datagen.flow_from_directory(validation_dir,
                                                             batch_size=32,
                                                             color_mode="rgb",
                                                             # shuffle = False,
                                                             target_size=(150, 150),  # ?
                                                             class_mode='categorical')

    history = model.fit(
        train_generator,
        steps_per_epoch=(train_len / 32),
        epochs=3,
        verbose=1,
        callbacks=[callbacks],
        validation_data=validation_generator,
        validation_steps=(val_len / 32)
    )

    # %matplotlib inline
    import matplotlib.pyplot as plt

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.figure()

    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Training and Validaion Loss')
    plt.figure()

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      batch_size=1,
                                                      target_size=(150, 150),
                                                      shuffle=False,
                                                      class_mode='categorical')

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    loss, acc = model.evaluate(test_generator, steps=(nb_samples), verbose=1)
    print('accuracy test: ', acc)

    model.save('model.h5')
else:
    print("Dataset found already downloaded!")

print('Model loaded!')
model_predict = load_model(resource_path('./model.h5'))
model_predict.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

print('Model compiled!')


def main():

    # Prompt the user to select a file
    file_paths = filedialog.askopenfilenames(title="Select Image Files", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

    if len(file_paths) == 0:
        Tk.messagebox.showinfo("No files selected", "You didn't select any files!")
        print("No files were selected!")
        return

    print(type(file_paths))

    # Now you have the file paths selected by the user, and you can proceed with further processing
    print("Selected files:", file_paths)
    display_images(file_paths)

    image_name = []
    image_conf = []
    predict_result = []

    for fn in file_paths:
        path = fn
        img = image.load_img(path, color_mode="rgb", target_size=(150, 150), interpolation="nearest")
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255

        images = np.vstack([img])
        classes = model_predict.predict(images, batch_size=10)

        max = np.amax(classes[0])
        if np.where(classes[0] == max)[0] == 0:
            image_name.append(fn)
            image_conf.append(max)
            predict_result.append('Fresh Apple')
        elif np.where(classes[0] == max)[0] == 1:
            image_name.append(fn)
            image_conf.append(max)
            predict_result.append('Fresh Banana')
        elif np.where(classes[0] == max)[0] == 2:
            image_name.append(fn)
            image_conf.append(max)
            predict_result.append('Fresh Orange')
        elif np.where(classes[0] == max)[0] == 3:
            image_name.append(fn)
            image_conf.append(max)
            predict_result.append('Rotten Apple')
        elif np.where(classes[0] == max)[0] == 4:
            image_name.append(fn)
            image_conf.append(max)
            predict_result.append('Rotten Banana')
        else:
            image_name.append(fn)
            image_conf.append(max)
            predict_result.append('Rotten orange')

    plt.figure(figsize=(15, 4))
    for n in range(len(image_name)):
        plt.subplot((int)(len(image_name) / 4) + 1, 4, n + 1)
        plt.subplots_adjust(hspace=0)
        plt.imshow(image.load_img(image_name[n], color_mode="rgb", target_size=(150, 150), interpolation="nearest"))
        title = f"Prediction: {predict_result[n]} ({round(float(image_conf[n]) * 100, 2)}%)"
        plt.title(title, color='black')
        plt.axis('off')
    plt.show()

    def predict():
        return predict_result

    for _fn_ in image_name:
        os.system(f'del {_fn_}')


current_mode = "dark"

customtkinter.set_appearance_mode(current_mode)
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("720x405")

background_image_dark = Tk.PhotoImage(file=resource_path("./abo bg dark.png"), format="PNG")
background_image_light = Tk.PhotoImage(file=resource_path("./abo bg.png"), format="PNG")
background_label = Tk.Label(master=root, image=background_image_dark)
background_label.place(relwidth=1, relheight=1)


def resize_image(image, max_width, max_height):
    # Resize the image while preserving aspect ratio
    width, height = image.size
    if width > max_width or height > max_height:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        return image.resize((new_width, new_height))
    else:
        return image


def display_images(file_paths):
    # Create a new window to display the images
    display_window = Tk.Toplevel(root)
    display_window.title("Selected Images")

    # Define maximum width and height for resizing images
    max_width = 500
    max_height = 500

    # Iterate over the selected file paths and display each image
    for idx, file_path in enumerate(file_paths):
        # Open image using PIL
        image = Image.open(file_path)

        # Resize the image
        resized_image = resize_image(image, max_width, max_height)

        # Convert image to Tkinter-compatible format
        tk_image = ImageTk.PhotoImage(resized_image)

        # Create a label to display the image
        label = Tk.Label(display_window, image=tk_image)
        label.image = tk_image  # Keep a reference to the image to prevent garbage collection
        label.grid(row=idx // 2, column=idx % 2, padx=10, pady=10)

    display_window.wait_window()


def toggle_mode():
    global current_mode
    if current_mode == "light":
        current_mode = "dark"
        customtkinter.set_appearance_mode(current_mode)

        background_label.configure(image=background_image_dark)

        mode_button.configure(text="Light Mode")

    else:
        current_mode = "light"
        customtkinter.set_appearance_mode(current_mode)

        background_label.configure(image=background_image_light)

        mode_button.configure(text="Dark Mode")
    root.update()


mode_button = customtkinter.CTkButton(master=root, text="Light Mode", command=toggle_mode, fg_color="transparent",
                                      border_width=2, text_color=("gray10", "#DCE4EE"))
# (master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
mode_button.pack(pady=10, padx=(10, 60), anchor="ne")

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=(0, 5), fill="both", padx=60, expand=True)

root.title("Detect your fruit is fresh or rotten!")

label = customtkinter.CTkLabel(master=frame, text="Welcome to the Fruit Classifier", font=("Roboto", 32))
label.pack(pady=12, padx=10)

select_image = customtkinter.CTkButton(master=frame, text="Pick Apple/Banana/Orange", command=main, border_width=2,
                                       border_color=("gray10", "#DCE4EE"))
select_image.pack(pady=12, padx=20, fill="x", expand=True, side="top", anchor="center", ipadx=20, ipady=10)

slider_progressbar_frame = customtkinter.CTkFrame(master=frame, fg_color="transparent")
slider_progressbar_frame.pack(pady=0, padx=10)
progressbar_1 = customtkinter.CTkProgressBar(slider_progressbar_frame, mode="indeterminnate")
progressbar_1.pack(pady=(0, 4), padx=20, fill="both", expand=True)
progressbar_1.start()

quit = customtkinter.CTkButton(master=frame, text="Quit", command=root.quit, fg_color=("red", "#DB1F48"))
quit.pack(pady=12, padx=10, ipadx=10, ipady=10)

developed_by = customtkinter.CTkLabel(master=root,
                                      text="Developed by- M Kavya Sree, Deepika Divya, Ayushi Raj & Anubhav Tekriwal",
                                      font=("Roboto", 12))
developed_by.pack(pady=(0, 5), padx=15, anchor="s")

root.mainloop()