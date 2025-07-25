import json
import os

def convert_to_lavis_coco_format(input_json_path, output_json_path, image_root):

    # Load the existing annotations (HuggingFace format or COCO-style annotations)
    with open(input_json_path, "r") as f:
        coco_data = json.load(f)

    # Map image_id to file_name (from the "images" section)
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    # Create a list of samples
    samples = []
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        
        if image_id not in image_id_to_filename:
            continue  # Skip if the image_id doesn't have a matching file_name

        # Now we get the correct filename for this image_id
        image_filename = image_id_to_filename[image_id]
        image_path = os.path.join(image_root, image_filename)  # Join with image_root to get the full path
        caption = ann["caption"]  # Caption from the annotation

        # Append the sample to the list
        samples.append({
            "image": image_path,
            "caption": caption,
            "image_id": image_id
        })

    # Write the converted data to the output file
    with open(output_json_path, "w") as f:
        json.dump({"samples": samples}, f, indent=2)

    print(f"✅ Converted and saved: {output_json_path}")


if __name__=='__main__':
    with open("/home/username/MSCoco_caption_gen/annotations/Lavis_captions_train2017.json", "r") as f:
        df= json.load(f)
        print(len(df))

    # convert_to_lavis_coco_format("/home/username/MSCoco_caption_gen/annotations/captions_val2017.json", 
    #                              "/home/username/MSCoco_caption_gen/annotations/Lavis_captions_val2017.json",
    #                              "/home/username/MSCoco_caption_gen/val2017/")
