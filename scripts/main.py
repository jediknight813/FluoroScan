import gradio as gr
from ultralytics import YOLO


# a simple gradio app to test out the model once it's finished training.s
def detect(modelFile, testImage):
    model = YOLO(modelFile)
    results = model([testImage])

    for result in results:
        boxes = result.boxes
        masks = result.masks
        keypoints = result.keypoints
        probs = result.probs
        result.save(filename='result.jpg')

    return "./result.jpg"


demo = gr.Interface(
    fn=detect,
    inputs=["file", "image"],
    outputs=["image"],
    title="Brightest Bio disease detection test"
)


if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860)
