from PIL import Image, ImageDraw
import random


# creates a bunch of images and labels them for yolov8 training, there way no way I was going to manually label every training image, I love python.
def create_training_images(training_types):
    for index, training_type in enumerate(training_types):
        for i in range(0, 300):
            filename = str(i)+"_"+training_type["name"].replace(" ", "_")
            image, bbox_center_x, bbox_center_y, bbox_width, bbox_height = create_dataset_image(training_type["color"])
            image.save("./training_data/images/"+filename+".jpeg")
            with open("./training_data/labels/"+filename+".txt", 'w') as file:
                file.write(f"{index} {bbox_center_x} {bbox_center_y} {bbox_width} {bbox_height}")


def create_dataset_image(color):
    image = Image.new('RGB', (512, 512), color='black')
    draw = ImageDraw.Draw(image)
    draw.ellipse((0, 0, 512, 512), 'white', 360)
    x = random.uniform(80, 400)
    y = random.uniform(80, 400)
    draw.ellipse((x-8, y-8, x+8, y+8), color, 360)
    bbox_x1 = x - 15
    bbox_y1 = y - 15
    bbox_x2 = x + 15
    bbox_y2 = y + 15
    center_x = (bbox_x1 + bbox_x2) / 2.0
    center_y = (bbox_y1 + bbox_y2) / 2.0
    center_x_norm = center_x / 512
    center_y_norm = center_y / 512
    bbox_width = bbox_x2 - bbox_x1
    bbox_height = bbox_y2 - bbox_y1
    bbox_width_norm = bbox_width / 512
    bbox_height_norm = bbox_height / 512
    return image, center_x_norm, center_y_norm, bbox_width_norm, bbox_height_norm


# creates the training dataset and labels.
create_training_images(
    [
        {"name": "Alzheimers Disease", "color": "red"},
        {"name": "Kidney Disease", "color": "blue"},
        {"name": "Osteoporosis", "color": "green"},
        {"name": "Lung Cancer", "color": "purple"}
    ]
)
