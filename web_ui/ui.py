import gradio as gr
from main import run_flow

demo = gr.Interface(fn=run_flow, 
            inputs=gr.Image(type="filepath"),
            outputs=[
                gr.Label(label="Predictions", num_top_classes=5), 
                gr.Label(label="What's in this image?",num_top_classes=10), 
                gr.Label(label="Landmarks",num_top_classes=5),
                gr.Textbox(label="Text found in image"),
                gr.Gallery(label="Similar images found online").style(grid=4),
                gr.Textbox(label="Web entities metadata")
            ],
            examples=["car_image.jpg"])
demo.launch(share=False, debug=False, max_threads=8)