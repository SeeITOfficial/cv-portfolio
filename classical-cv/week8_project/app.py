import gradio as gr
import cv2 as cv
from operations import (to_grayscale, apply_blur, apply_canny, apply_clahe, show_colorspaces, plot_histogram, apply_sharpening, apply_backprojection, detect_contours)

def process(img, operation, blur_strength, canny1, canny2, sharpen_amount, clahe_gridsize, contour_area):
    if img is None:
        return None, None, "Upload an image first!"
    
    bgr_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    extra_output = None
    info_text = ""

    if operation == "Grayscale":
        result = to_grayscale(bgr_img)
    
    elif operation == "Blur":
        result = apply_blur(bgr_img, int(blur_strength))

    elif operation == "Canny Edge Detection":
        result = apply_canny(bgr_img, int(canny1), int(canny2))

    elif operation == "CLAHE":
        result = apply_clahe(bgr_img, int(clahe_gridsize))
    
    elif operation == "Colorspaces (HSV & LAB)":
        hsv, lab = show_colorspaces(bgr_img)
        result      = hsv
        extra_output = lab
        info_text   = "Left Output: HSV -> bgr_img | Right Output: LAB -> bgr_img"

    elif operation == "Histogram":
        result = plot_histogram(bgr_img)
        # result is already RGB numpy array from matplotlib
        return result, None, "bgr_img channel histograms"

    elif operation == "Sharpening":
        result = apply_sharpening(bgr_img, float(sharpen_amount))

    elif operation == "Contour Detection":
        result, count = detect_contours(bgr_img, int(canny1), int(canny2), int(blur_strength), int(contour_area))
        info_text = f"Contours detected: {count}"

    else:
        result = bgr_img
        info_text = "Select an operation."

    rgb_result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
    if extra_output is not None and extra_output.size > 0:
        extra_rgb = cv.cvtColor(extra_output, cv.COLOR_BGR2RGB)
    else:
        extra_rgb = None
        
    return rgb_result, extra_rgb, info_text


with gr.Blocks(title="CV Image Processing Toolkit") as app:
    gr.Markdown("# CV Image Processing Toolkit")
    gr.Markdown("Upload an image and select an operation.")

    #ROW 1: operation selector
    operation = gr.Dropdown(
        choices=[
            "Grayscale", "Blur", "Canny Edge Detection",
            "CLAHE", "Colorspaces (HSV & LAB)", "Histogram",
            "Sharpening", "Contour Detection"
        ],
        label="Operation",
        value="Grayscale"
    )

    #ROW 2: sliders horizontal
    with gr.Row():
        blur_strength  = gr.Slider(1,   25,  value=5,   step=1,   label="Blur Strength")
        canny_t1       = gr.Slider(0,   500, value=50,  step=1,  label="Canny Threshold 1")
        canny_t2       = gr.Slider(0,   500, value=150, step=1,  label="Canny Threshold 2")
        sharpen_amount = gr.Slider(0.5, 5.0, value=1.5, step=0.5, label="Sharpen Amount")
        clahe_gridsize = gr.Slider(1,   16,  value=8,   step=1,   label="CLAHE Grid Size")
        contour_area   = gr.Slider(1, 1000, value=1, step=1, label="Conntour Area")

    #ROW 3: input and output images
    with gr.Row():
        input_img  = gr.Image(label="Input Image", type="numpy")
        output_img = gr.Image(label="Result")
        output_img2 = gr.Image(label="Secondary Output")

    info_box = gr.Textbox(label="Info")

    #LIVE UPDATES
    all_inputs = [input_img, operation, blur_strength, canny_t1, canny_t2, sharpen_amount, clahe_gridsize, contour_area]
    all_outputs = [output_img, output_img2, info_box]

    for component in [operation, blur_strength, canny_t1, canny_t2, sharpen_amount, clahe_gridsize, input_img, contour_area]:
        component.change(fn=process, inputs=all_inputs, outputs=all_outputs)


app.launch(server_name="0.0.0.0", server_port=7860)