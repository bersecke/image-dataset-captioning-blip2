# Image Dataset Captioning with BLIP2

This repository can be used to generate descriptive text prompts for images in a dataset, which can later be useful for training generative models with text conditioning or other tasks. The BLIP2 model from the [LAVIS library](https://github.com/salesforce/LAVIS) is utilized for this purpose. It is presented with questions that cover the aspects of interest, it provides answers to these, and said answers are then interpreted and combined into full text prompts.

## Installation

Requirements can be installed from the dedicated file included in the repository:
```bash
pip install -r requirements.txt
```

## Usage

The generation of text prompts for a given dataset can be run with `image_captioning.py`. The script should be provided the following arguments:
* `--model_name`: Name of the BLIP2 model to be used from the model zoo in the LAVIS library. If none is provided, `"blip2_t5"` is used by default.
* `--model_type`: Name of the BLIP2 model type to be used from the model zoo in the LAVIS library. If none is provided, `"pretrain_flant5xxl"` is used by default.
* `--data_dir`: Path to the directory holding the targeted dataset.
* `--dataset_format`: String indicating the name of the dataset being targeted. At the moment only `"cityscapes"` and `"mapillary"` are supported. Readers for additional datasets can be added in `utils/file_listing.py`.
* `--sets`: String indicating the sets from the selected dataset to be targeted (e.g. `"train"`, `"val"` or `"test"`, in the case of Cityscapes). If several sets are relevant these can be combined into the same string with "-" as a spacer (e.g. `"train-val-test"` for Cityscapes).
* `--question_prompts`: Path to the file detailing which questions the model should answer for each class of interest.
* `--semantics_config` (Optional): Path to config with semantic label information for the dataset (class numbering). If provided, questions for classes that have very little presence in a particular image are skipped.
* `--results_dir`: Path to the directory where to save resulting text prompts associated to the paths of the images these relate to.

### Preparing custom question prompts file

The content of the `.yaml` file for the `--question_prompts` argument should be structured in dictionary form as follows:

```yaml
class_name_1:
    - question_1.1: null
    - question_1.2:
        - term_for_yes_answer
        - term_for_no_answer
    - ...
class_name_2:
    - question_2.1: null
    - question_2.2: null
    - ...
...
```
If a file is provided for the `--semantics_config` argument, the class names should be the same as the ones listed in said file. This allows for the script to check whether each of these classes has a significant presence in an image, and if that is not the case, it skips their respective questions. The order of the classes and questions in the configuration file determines the order in which these are added to the combined text prompt.

Questions can either be open-ended or "yes or no" questions. Open ended questions (e.g. `question_1.1`) should be provided a `null` value, while "yes or no" questions (e.g. `question_1.2`) should be provided a list of two strings: one for how "yes" answers should be interpreted and another for how "no" answers should be interpreted.


Note that examples for all the described arguments and associated configuration files can be found for the Cityscapes example explained further down.

### Running Cityscapes example

An bash script example `caption_cityscapes.sh` on how to run the functionality on the Cityscapes dataset is provided in the repository. The only argument that needs adjusting is the `--data_dir`, which should be set to the path where the user keeps the Cityscapes dataset.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/) 
