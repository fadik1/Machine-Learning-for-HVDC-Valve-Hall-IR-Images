import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog, StringVar
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KernelDensity
import os

cnn_path = 'C:\\Users\\fadip\\.spyder-py3\\imported_colab_models\\93'         # insert model path
cnn_model = tf.keras.models.load_model(cnn_path)
Classes = ["idle", "gpu_stress", "cpu_stress", "cpu_gpu_stress"]                #list classes names
Classes.sort() #to match predictions to classes list, you must sort by alphabetical order

# cpu_stress_autoencoder_path = 'C:\\Users\\fadip\\.spyder-py3\\imported_colab_models\\cpu_stress_autoencoder'         # insert model path
# cpu_stress_autoencoder_model = tf.keras.models.load_model(cpu_stress_autoencoder_path)                               #to match predictions to classes list, you must sort by alphabetical order
cpu_stress_encoder_path = 'C:\\Users\\fadip\\.spyder-py3\\imported_colab_models\\cpu_stress_encoder'         # insert model path
cpu_stress_encoder_model = tf.keras.models.load_model(cpu_stress_encoder_path)                               #to match predictions to classes list, you must sort by alphabetical order

cpu_encoded_images_flat = np.genfromtxt('C:\\Users\\fadip\\.spyder-py3\\imported_colab_models\\cpu_encoded_images_flat.csv', delimiter=',')
cpu_val_enc_flat = np.genfromtxt('C:\\Users\\fadip\\.spyder-py3\\imported_colab_models\\cpu_val_enc_flat.csv', delimiter=',')
cpu_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(cpu_encoded_images_flat) 
cpu_validation_density_scores = cpu_kde.score_samples(cpu_val_enc_flat)
cpu_min_quantile = np.quantile(cpu_validation_density_scores, 0.01)   #obtain 0.01 smallest value on validatation density score
cpu_max_quantile = np.quantile(cpu_validation_density_scores, 0.99)   #obtain 0.99 biggest value on validatation density score
    

cpu_min_quantile_threshold= cpu_min_quantile - 50         #change 10 to a number based on tolerance level
cpu_max_quantile_threshold= cpu_max_quantile + 50         #change 10 to a number based on tolerance level
    

# gpu_stress_autoencoder_path = 'C:\\Users\\fadip\\.spyder-py3\\imported_colab_models\\gpu_stress_autoencoder'         # insert model path
# gpu_stress_autoencoder_model = tf.keras.models.load_model(gpu_stress_autoencoder_path)                               #to match predictions to classes list, you must sort by alphabetical order
gpu_stress_encoder_path = 'C:\\Users\\fadip\\.spyder-py3\\imported_colab_models\\gpu_stress_encoder'         # insert model path
gpu_stress_encoder_model = tf.keras.models.load_model(gpu_stress_encoder_path)                               #to match predictions to classes list, you must sort by alphabetical order

gpu_encoded_images_flat = np.genfromtxt('C:\\Users\\fadip\\.spyder-py3\\imported_colab_models\\gpu_encoded_images_flat.csv', delimiter=',')
gpu_val_enc_flat = np.genfromtxt('C:\\Users\\fadip\\.spyder-py3\\imported_colab_models\\gpu_val_enc_flat.csv', delimiter=',')
gpu_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(gpu_encoded_images_flat) 
gpu_validation_density_scores = gpu_kde.score_samples(gpu_val_enc_flat)
gpu_min_quantile = np.quantile(gpu_validation_density_scores, 0.01)   #obtain 0.01 smallest value on validatation density score
gpu_max_quantile = np.quantile(gpu_validation_density_scores, 0.99)

gpu_min_quantile_threshold= gpu_min_quantile - 50          #change 10 to a number based on tolerance level
gpu_max_quantile_threshold= gpu_max_quantile + 50         #change 10 to a number based on tolerance level


def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        print(img_display)
        img_display.pack_forget()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                        filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 300
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()
    

    
def classify():
    IMG_width = 150                                                                 # specify image width
    IMG_height= 150                                                                 # specify image height

    original = Image.open(image_data)                                           #open image
    resized_image_data = np.array(original.resize((IMG_height, IMG_width), Image.ANTIALIAS))    # specify the original model image input. in this case, it's 150*150
    resized_image_data = resized_image_data / 255
    image_batch = np.expand_dims(resized_image_data, axis=0)
    
    pred = cnn_model.predict(image_batch)
    print(pred)
    score = tf.nn.softmax(pred[0])                                                       #use only if model last layer is dense layer
    for i in range(0, len(score)):
         tk.Label(frame,
                    text= str(Classes[i]).upper() + ': ' + str(round(float(score[i])*100, 3)) + '%').pack()

def cpu_change_detect():
    # reconstruction_error_threshold = 0.020
    
    layer=cpu_stress_encoder_model.get_layer('encoder_maxpool_4')
    layer_list=layer.get_output_at(0).get_shape().as_list()
    dim= layer_list[1]*layer_list[2]*layer_list[3]                                  # dim is equal for cpu and gpu
    IMG_width = 160                                                                # specify image width
    IMG_height= 160   
    
    original = Image.open(image_data) 
    w, h = original.size
    left=3*w/15                           #specify the point of crop. Go from (0,0) to (left, upper). height is 6, width is 8. cpu dimensions
    upper=1.5*h/11
    right=6*w/15                           #specify the point of crop. Go from (image size,image size) to (right, lower). 7*w/8 
    lower=4.5*h/11                    # 5*h/6
    imCrop = original.crop((left, upper, right, lower)) #crop
    
    resized_image_data = np.array(imCrop.resize((IMG_height, IMG_width), Image.ANTIALIAS))
    resized_image_data = resized_image_data / 255
    image_batch = tf.expand_dims(resized_image_data, 0)

    encoded_img = cpu_stress_encoder_model.predict(image_batch) # Create a compressed version of the image using the encoder
    encoded_img = [np.reshape(image_batch, (dim)) for image_batch in encoded_img] # Flatten the compressed image

    density = cpu_kde.score_samples(encoded_img)[0] # get a density score for the new image
    print(f'density: {density}')

    if density < cpu_min_quantile_threshold or density > cpu_max_quantile_threshold:# or reconstruction_error > reconstruction_error_threshold:
        cpu_label=tk.Label(frame,
                    text= str("CPU Issue Detected").upper()).pack()
    else:
        cpu_label=tk.Label(frame,
                    text= str("CPU Issue Not Detected").upper()).pack()
        
def gpu_change_detect():
    # reconstruction_error_threshold = 0.020

    layer=gpu_stress_encoder_model.get_layer('encoder_maxpool_4')
    layer_list=layer.get_output_at(0).get_shape().as_list()
    dim= layer_list[1]*layer_list[2]*layer_list[3]                                  # dim is equal for cpu and gpu
    
    IMG_width = 160                                                                # specify image width
    IMG_height= 160   
    
    original = Image.open(image_data) 
    w, h = original.size
    left=6*w/15                           #specify the point of crop. Go from (0,0) to (left, upper). height is 6, width is 8. cpu dimensions
    upper=7.5*h/11
    right=10.5*w/15                           #specify the point of crop. Go from (image size,image size) to (right, lower). 7*w/8 
    lower=11*h/11                    # 5*h/6
    imCrop = original.crop((left, upper, right, lower)) #crop
    
    resized_image_data = np.array(imCrop.resize((IMG_height, IMG_width), Image.ANTIALIAS))
    resized_image_data = resized_image_data / 255
    image_batch = tf.expand_dims(resized_image_data, 0)
    
    encoded_img = gpu_stress_encoder_model.predict(image_batch) # Create a compressed version of the image using the encoder
    encoded_img = [np.reshape(image_batch, (dim)) for image_batch in encoded_img] # Flatten the compressed image

    density = gpu_kde.score_samples(encoded_img)[0] # get a density score for the new image
    print(f'density: {density}')


    
    if density < gpu_min_quantile_threshold or density > gpu_max_quantile_threshold:
        # l1 = tk.Label(frame,
        #              text= str("GPU Issue Detected").upper()).pack()
        l1.config(text='hello')

    else:
        # l1 = tk.Label(frame,
        #              text= str("GPU Issue Not Detected").upper()).pack()
        l1.config(text='hii')
    

def assist_me():
    os.startfile('ENEL503-W21-Midterm.pdf')
 
# def press():
#     l1.config(text="hello")
    
root = tk.Tk()
root.title('Portable Image Classifier')
root.iconbitmap('C:\\Users\\fadip\\.spyder-py3\\imported_colab_models\\icon3.ico')
root.resizable(False, False)
help_button = tk.Button(root, text='Help',
                        padx=15, pady=0, command=assist_me)
help_button.pack(anchor="ne", side=tk.TOP)
tit = tk.Label(root, text="Altalink Thermal Image Classifier", padx=25, pady=6, font=("", 11)).pack()

canvas = tk.Canvas(root, height=500, width=550, bg='grey')
canvas.pack()

frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

choose_image = tk.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=load_img)
choose_image.pack(side=tk.LEFT)

cnn_image = tk.Button(root, text='Classify Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=classify)
cnn_image.pack(side=tk.RIGHT)

encoder_image = tk.Button(root, text='Detect CPU Change',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=cpu_change_detect)
encoder_image.pack(side=tk.RIGHT)

encoder_image = tk.Button(root, text='Detect GPU Change',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=gpu_change_detect)
encoder_image.pack(side=tk.RIGHT)

l1 = tk.Label(frame, text="hi")
l1.pack()


root.mainloop()