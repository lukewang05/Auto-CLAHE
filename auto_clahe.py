# Auto-CLAHE Program
# Marcus M. Hansen and Ainiu L. Wang

import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import requests
from pixstem.api import PixelatedSTEM
from hyperspy.api import load
import hyperspy.api as hs
from hyperspy import io_plugins
from multiprocessing import Pool
import tqdm

# declaring global variables used between functions
file = None
selected_points = None
input_file_path = None


# prompts file dialog for user to select file
def load_file():
    global file, input_file_path
    label3['text'] = "Loading file...\n"
    input_file_path = filedialog.askopenfilename()
    root.update()

    # Loading file and error catching
    try:
        file = PixelatedSTEM(load(input_file_path))
        label3['text'] = label3['text'] + "File loaded.\n"
    except ValueError:
        label3['text'] = label3['text'] + "Please select a file and try again.\n"
    except OSError:
        label3['text'] = label3['text'] + "Error loading. Please check the file path and try again.\n"


# creates the virtual bright-field image to navigate the dataset
def create_bright_field_image(stem_file):
    # creates equal sized # of sections to take the center of the image
    sections = 2
    image_length = len(stem_file.data[0][0])
    section_size = image_length / sections
    section1 = int((image_length / 2) - (section_size / 2))
    section2 = int((image_length / 2) + (section_size / 2))

    # Creates the bright_field image by slicing the center section of all images in the block file and averaging it for
    # their respective pixel value in the 4D array.
    bright_field_img = [[]]
    temp_array = None
    for i in tqdm.tqdm(range(len(stem_file.data))):
        for j in range(len(stem_file.data[i])):
            # creates a horizontal slice of the image and applies auto contrast to create a more distinct image
            # Auto contrast significantly increases computational time
            temp_img = Image.fromarray(np.array(stem_file.data[i][j], dtype='uint8'))
            temp_img = ImageOps.autocontrast(temp_img, cutoff=1)

            temp_slice = np.array(temp_img)[section1:section2]

            # refines to slice to be a square
            for r in range(len(temp_slice)):
                temp_array = temp_slice[r][section1:section2]

            # takes the average value of the pixels in the slice as adds them to the bright_field image
            bright_field_img[i].append(int(np.round(np.mean(np.asarray(temp_array)))))
        if i != len(stem_file.data) - 1:
            bright_field_img.append([])
    bright_field_img = np.array(bright_field_img, dtype='uint8')
    bright_field_img = 255 - bright_field_img
    bright_field_img = Image.fromarray(bright_field_img)
    bright_field_img_arr = np.array(bright_field_img)
    bright_field_img.save('virtual bright_field image.jpeg', format='jpeg')
    return bright_field_img_arr


# Auto-CLAHE method
# no user-input parameter
def auto_clahe(image):
    avg_val = np.average(image)
    clip_lim = round(10.0 / avg_val)
    clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=(4, 4))
    image = clahe.apply(image)
    return image


# Main Analysis window
def start_analysis():
    global file, selected_points
    label3['text'] = label3['text'] + "Generating virtual bright-field image...\n"
    frame.update()
    if file is not None:
        # Previews the diffraction pattern at the selected point using the given parameters
        def preview_point(point):
            # assigning user parameters to variables
            print(point)

            # copies the point data to avoid altering the main data
            img_point = file.data[point[1]][point[0]].copy()
            filter_preview_img = auto_clahe(img_point)

            # If the data is an array, converts it to a PIL Image due to potential conflicts with blob RGB and Grayscale
            if isinstance(filter_preview_img, np.ndarray):
                filter_preview_img = Image.fromarray(filter_preview_img)

            # resizes the image to fit the canvas and replace the current image with an updated filtered image
            filter_preview_img = filter_preview_img.resize((400, 400))
            filter_preview_img = ImageTk.PhotoImage(image=filter_preview_img)
            r.filter_preview_img = filter_preview_img
            filtered_canvas.itemconfigure(filtered_img_preview, image=filter_preview_img)

        def get_mouse_xy(event):
            global selected_points, file
            nonlocal bright_field_img_arr, img_x, img_y

            # get the mouse click position depending on the image shape due to resize scaling in rectangular images
            if img_x > img_y:
                point = (int(event.x * img_x / 400), int(event.y * img_x / 400))
            elif img_x < img_y:
                point = (int(event.x * img_y / 400), int(event.y * img_y / 400))
            else:
                point = (int(event.x * img_x / 400), int(event.y * img_y / 400))

            # displays selected diffraction pattern from .blo file
            preview_img = np.asarray(file.data[point[1]][point[0]])
            preview_img = Image.fromarray(preview_img).resize((400, 400))
            preview_img = ImageTk.PhotoImage(image=preview_img)
            r.preview_img = preview_img
            c2.itemconfigure(point_img, image=preview_img)

            r.point = point
            preview_point(point)
            r.update()

        # processes the entire dataset for filtering with the given user parameters
        def filter_file():
            global file, input_file_path

            file_array = np.zeros(np.array(file.data).shape, dtype='uint8')

            # create a flattened copy of the data for multiprocessing
            multiprocessing_list = []
            for y in range(len(file.data)):
                for x in range(len(file.data[y])):
                    multiprocessing_list.append(file.data[y][x])

            results = []
            pool = Pool(processes=None)
            # runs the desired filtering method on all the images in the array
            # Processes fast but uses a lot of memory, can remove multiprocessing for reduced memory usage at the
            # cost of speed
            for output in tqdm.tqdm(pool.imap(auto_clahe, multiprocessing_list),
                                    total=len(multiprocessing_list)):
                results.append(output)
                pass
            pool.close()

            i = 0
            # reshapes the array back into the original shape from the flattened results
            for row in range(len(file_array)):
                for col in range(len(file_array[row])):
                    file_array[row][col] = results[i]
                    i += 1
            stem_file_array = hs.signals.Signal2D(file_array)
            # saves the file with the original name plus suffixes based on the user parameters
            file_name = input_file_path[:-4] + '_Auto_CLAHE'
            io_plugins.blockfile.file_writer(file_name + '.blo', stem_file_array)
            label3['text'] = label3['text'] + "Filtered file saved.\n"
            return

        # main window
        r = tk.Toplevel(root)
        r.title('')

        canvas_height = 620
        canvas_width = 1360
        c = tk.Canvas(r, height=canvas_height, width=canvas_width)
        c.pack()

        f = tk.Frame(r, bg='#FFFFFF')
        f.place(relwidth=1, relheight=1)

        bright_field_img_arr = create_bright_field_image(file)
        img_x = len(bright_field_img_arr[0])
        img_y = len(bright_field_img_arr)
        # adjusts the image size to scale up to 400 based on the aspect ratio of the virtual bright-field image.
        if img_x > img_y:
            tk_image = Image.fromarray(bright_field_img_arr).resize((400, int((img_y / img_x) * 400)))
        elif img_x < img_y:
            tk_image = Image.fromarray(bright_field_img_arr).resize((int((img_x / img_y) * 400), 400))
        else:
            tk_image = Image.fromarray(bright_field_img_arr).resize((400, 400))

        # canvas for the virtual bright_field image

        if img_x > img_y:
            c1 = tk.Canvas(r, width=400, height=int((img_y / img_x) * 400))
        elif img_x < img_y:
            c1 = tk.Canvas(r, width=int((img_x / img_y) * 400), height=400)
        else:
            c1 = tk.Canvas(r, width=400, height=400)

        c1.place(relx=0.05, anchor='nw')
        tk_image = ImageTk.PhotoImage(image=tk_image)
        c1.create_image(0, 0, anchor='nw', image=tk_image)
        c1.bind('<Button-1>', get_mouse_xy)

        # canvas for preview diffraction pattern
        c2 = tk.Canvas(r, width=400, height=400)
        c2.place(relx=0.5, anchor='n')
        point_img = c2.create_image(0, 0, anchor='nw', image=None)

        filtered_canvas = tk.Canvas(r, width=400, height=400)
        filtered_canvas.place(relx=0.95, anchor='ne')
        filtered_img_preview = filtered_canvas.create_image(0, 0, anchor='nw', image=None)

        # Image texts
        bright_field_text = tk.Label(f, text='Virtual Bright-Field Image', bg='#FFFFFF', font=('Calibri', 20),
                                     fg='#373737')
        bright_field_text.place(relx=0.09, rely=0.55)

        orig_text = tk.Label(f, text='Original Diffraction Pattern', bg='#FFFFFF', font=('Calibri', 20), fg='#373737')
        orig_text.place(relx=0.39, rely=0.55)

        filtered_text = tk.Label(f, text='Filtered Diffraction Pattern', bg='#FFFFFF', font=('Calibri', 20),
                                 fg='#373737')
        filtered_text.place(relx=0.7, rely=0.55)

        # Filter and Export File button
        export_button = tk.Button(f, text='Filter and Export File', bg='#F3F3F3', font=('Calibri', 20),
                                  highlightthickness=0, bd=0, activebackground='#D4D4D4', activeforeground='#252525',
                                  command=lambda: filter_file(), pady=0.02, fg='#373737', borderwidth='2',
                                  relief="groove")
        export_button.place(relx=0.35, rely=0.75, relwidth=0.30, relheight=0.10)

        r.mainloop()

    else:
        label3['text'] = "Please select a file and try again.\n"


if __name__ == "__main__":
    HEIGHT = 540
    WIDTH = 800

    root = tk.Tk()
    root.title('')

    canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
    canvas.pack()
    frame = tk.Frame(root, bg='#FFFFFF')
    frame.place(relwidth=1, relheight=1)

    # TAMU MSEN logo
    try:
        url = 'https://github.com/TAMU-Xie-Group/PED-Strain-Mapping/blob/main/msen.png?raw=true'
        msen_image = Image.open(requests.get(url, stream=True).raw)
        msen_image = msen_image.resize((200, 40))
        msen_image = ImageTk.PhotoImage(msen_image)
        label1 = tk.Label(frame, image=msen_image, bg='#FFFFFF')
        label1.place(relx=0.05, rely=0.05, anchor='w')
    except:
        print('Error: no internet connection for TAMU MSEN logo')

    # Menu Label
    label2 = tk.Label(frame, text='Auto-CLAHE', bg='#FFFFFF', font=('Times New Roman', 34),
                      fg='#373737')
    label2.place(relx=0.10, rely=0.12, relwidth=0.8, relheight=0.1)

    # Buttons
    button = tk.Button(frame, text='Load File', bg='#F3F3F3', font=('Calibri', 24), highlightthickness=0, bd=0,
                       activebackground='#D4D4D4', activeforeground='#252525',
                       command=lambda: load_file(), pady=0.02, fg='#373737', borderwidth='2',
                       relief="groove")
    button.place(relx=0.29, rely=0.27, relwidth=0.42, relheight=0.07)

    button1 = tk.Button(frame, text='Filtering Preview', bg='#F3F3F3', font=('Calibri', 24), highlightthickness=0, bd=0,
                        activebackground='#D4D4D4', activeforeground='#252525',
                        command=lambda: start_analysis(), pady=0.02, fg='#373737', borderwidth='2',
                        relief="groove")
    button1.place(relx=0.29, rely=0.36, relwidth=0.42, relheight=0.07)

    # Text Output box
    label3 = tk.Message(frame, bg='#F3F3F3', font=('Calibri', 15), anchor='nw', justify='left', highlightthickness=0,
                        bd=0, width=640, fg='#373737', borderwidth=2, relief="groove")
    label3['text'] = "This program was designed by Marcus M. Hansen and Ainiu L. Wang.\n"
    label3.place(relx=0.1, rely=0.56, relwidth=0.8, relheight=0.32)

    root.mainloop()
