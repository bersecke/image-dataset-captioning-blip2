import os
import argparse
import yaml
import torch
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

from utils.data_processing import read_images, process_images, read_semantics
from utils.miscellaneous import check_present_classes, process_yesno_result

import importlib

file_listing = importlib.import_module(
    "realsim-24.utilities.data_utils.file_listing")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create image text prompt descriptions via BLIP2 inference."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="blip2_t5",
        help=
        "Name of the BLIP2 model to be used from the model zoo in the LAVIS library."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="pretrain_flant5xxl",
        help=
        "Name of the BLIP2 model type to be used from the model zoo in the LAVIS library."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory holding the targeted dataset.")
    parser.add_argument("--dataset_format",
                        type=str,
                        default="basic",
                        help="String indicating if the dataset: \
                - Only contains image data ('basic') \
                - Includes labels and is in the Mapillary folder structure \
                    ('mapillary') \
                - Includes labels and is in the Cityscapes folder structure \
                    ('cityscapes')")
    parser.add_argument(
        "--sets",
        required=True,
        type=str,
        help="String indicating the sets from the selected dataset to be \
            targeted.")
    parser.add_argument(
        "--question_prompts",
        required=True,
        type=str,
        help=
        "Path to the file detailing which questions the model should answer for each class of interest."
    )
    parser.add_argument("--label_config",
                        type=str,
                        default=None,
                        required=False,
                        help="Path to label config.")
    parser.add_argument("--label_mapping_config",
                        type=str,
                        default=None,
                        required=False,
                        help="Path to label mapping config.")
    parser.add_argument(
        "--results_dir",
        required=True,
        type=str,
        help=
        "Path to the directory where to save resulting text prompts associated to the paths of the images these relate \
            to.")

    args = parser.parse_args()

    return args


def main(args):
    # Establishing device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Preparing model and image processors
    print("Loading BLIP2 model...")
    model, vis_processors, _ = load_model_and_preprocess(
        name=args.model_name,
        model_type=args.model_type,
        is_eval=True,
        device=device)

    # Listing file names using the dedicated function from realsim-24
    image_path_list, _, semantics_path_list = file_listing.list_files(
        args.data_dir, args.dataset_format, args.sets, None, True)

    # Loading image data
    print("Loading images...")
    image_list = read_images(image_path_list)
    image_list = process_images(image_list, vis_processors, device)

    # Load semantic data
    if args.label_config:
        print("Loading semantic maps...")
        # Preparing label config
        with open(args.label_config, 'r') as f:
            label_name_info = yaml.load(f, Loader=yaml.SafeLoader)["labels"]
            label_name_info_inv = dict(
                (v, k) for k, v in label_name_info.items())

        # Preparing label mapping config
        # NOTE Might want to re-utilize this code on both this and the realsim-24 repos
        if args.label_mapping_config:
            with open(args.label_mapping_config, 'r') as f:
                map_info = yaml.load(f, Loader=yaml.SafeLoader)
                map_field_name = f"learning_map_{os.path.splitext(os.path.basename(args.label_config))[0]}"
                map_class_nums = (
                    torch.Tensor(list(map_info[map_field_name].keys())).to(
                        torch.int8),
                    torch.Tensor(list(map_info[map_field_name].values())).to(
                        torch.int8))
        else:
            map_class_nums = None

        # Loading semantic maps
        semantics_list = read_semantics(semantics_path_list, map_class_nums)
    else:
        semantics_list = [None] * len(image_list)

    # Reading question prompts config
    with open(args.question_prompts, 'r') as f:
        question_prompts_info = yaml.load(f, Loader=yaml.SafeLoader)

    # Prepare config where to save results
    os.makedirs(args.results_dir, exist_ok=True)
    summary_path = os.path.join(args.results_dir, "gen_txt_prompts.yml")

    # Inferring answers and building full text prompts per image
    results_dict = {}
    print("Generating text prompt descriptions...")
    for image, semantics, image_path in tqdm(
            zip(image_list, semantics_list, image_path_list)):
        image_prompt = ""

        # Check which classes actually show up in an image
        if args.label_config:
            present_classes = check_present_classes(
                semantics, question_prompts_info.keys(), label_name_info_inv)
        else:
            present_classes = question_prompts_info.keys()

        # for i, class_name in enumerate(question_prompts_info.keys()):
        for i, class_name in enumerate(present_classes):
            image_prompt += class_name
            for question in question_prompts_info[class_name]:
                result = model.generate(
                    {
                        "image": image,
                        "prompt": list(question.keys())[0]
                    },
                    # NOTE Consider making settings below into arguments too
                    use_nucleus_sampling=True,
                    num_captions=1,
                    max_length=4,
                    length_penalty=1,
                    repetition_penalty=1.5,
                    temperature=1)

                question_info = list(question.values())[0]
                if question_info:
                    processed_result = process_yesno_result(
                        result[0], question_info)
                else:
                    processed_result = result[0]

                if processed_result != "":
                    image_prompt += (" " + processed_result)
            if i < len(present_classes) - 1:
                image_prompt += ", "

        results_dict[image_path] = image_prompt

    with open(summary_path, "w") as f:
        yaml.dump(results_dict, f)

    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
