import os
import sys
import cv2
import torch
from model.utils import preprocess_image, load_model

model = load_model()


def is_forgery (image_path):
    image = preprocess_image(image_path)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        confidence = torch.softmax(output, dim=1).max().item()
        predicted_class = output.argmax(dim=1).item()

    is_forged = predicted_class == 1

    return is_forged, confidence


def process_folder(folder_path):
    out_file = "out.csv"

    with open(out_file, mode='w') as file:
        file.write('file_name,is_forged,confidence\n')
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
           
            try:
                is_forged, confidence = is_forgery(file_path)
                file.write(f'{file_name},{is_forged},{confidence}\n')

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print(f"Results saved to {out_file}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        sys.exit(1)

    process_folder(folder_path)

if __name__ == "__main__":
    main()
